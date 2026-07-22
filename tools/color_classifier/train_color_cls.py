# train_color_classifier.py
import os, glob, random, argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


class ColorMaskDataset(Dataset):
    def __init__(self, image_dir, txt_dir, transform=None):
        self.samples = []
        self.image_dir = image_dir
        self.transform = transform
        for txt_file in glob.glob(os.path.join(txt_dir, "*.txt")):
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            image_path = os.path.join(image_dir, base_name + ".jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, base_name + ".png")
                if not os.path.exists(image_path):
                    continue
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    color_flag = int(parts[-1])
                    if color_flag not in (0, 1):
                        continue
                    coords = list(map(float, parts[1:-1]))
                    polygon_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    self.samples.append((image_path, polygon_points, color_flag))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts, color_flag = self.samples[idx]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        poly = np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        x, y, ww, hh = cv2.boundingRect(poly)
        pad = int(max(ww, hh) * 0.1) + 1
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + ww + pad, w), min(y + hh + pad, h)

        crop = img[y0:y1, x0:x1]
        m_crop = mask[y0:y1, x0:x1]
        crop_masked = cv2.bitwise_and(crop, crop, mask=(m_crop > 0).astype('uint8') * 255)

        crop_rgb = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2RGB)
        if self.transform:
            crop_rgb = self.transform(crop_rgb)
        else:
            default_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            crop_rgb = default_transform(crop_rgb)
        return crop_rgb, torch.tensor(color_flag, dtype=torch.long)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ColorMaskDataset(args.image_dir, args.txt_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty! Check image/txt paths and label format.")

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_model)
            print(f"Saved best model: {args.output_model}")

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Train ResNet18 Color Classifier (0=colorless, 1=colored)")
    parser.add_argument("--image-dir", type=str, default=r"datasets/CTG2.0/train/images",
                        help="Directory of training images")
    parser.add_argument("--txt-dir", type=str, required=True,
                        help="Directory of label files with color flag (REQUIRED)")
    parser.add_argument("--output-model", type=str, default="color_classifier_resnet18.pth",
                        help="Output path for trained weight file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
