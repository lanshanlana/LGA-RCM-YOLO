# eval_color_classifier.py
import os, glob, random, argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


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
        return crop_rgb, torch.tensor(color_flag, dtype=torch.long), os.path.basename(img_path)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ColorMaskDataset(args.image_dir, args.txt_dir, transform)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty! Check image/txt paths and label format.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, filenames = [], [], []
    with torch.no_grad():
        for images, labels, file_names in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(labels.numpy())
            filenames.extend(file_names)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['colorless', 'colored'], digits=4))

    confusion_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['colorless', 'colored'],
                yticklabels=['colorless', 'colored'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of color_classifier_resnet18')
    plt.tight_layout()
    plt.savefig(args.cm_output_path, dpi=300)
    plt.show()

    os.makedirs(args.error_sample_dir, exist_ok=True)
    for true_label, pred_label, fn in zip(y_true, y_pred, filenames):
        if true_label != pred_label:
            source_path = os.path.join(args.image_dir, fn)
            dst_path = os.path.join(args.error_sample_dir, f"{fn}_true{true_label}_pred{pred_label}.jpg")
            if os.path.exists(source_path):
                if os.name == "nt":
                    os.system(f'copy "{source_path}" "{dst_path}" >nul')
                else:
                    os.system(f'cp "{source_path}" "{dst_path}"')

    print(f"\nConfusion matrix saved to {args.cm_output_path}")
    print(f"Misclassified samples saved to {args.error_sample_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 Color Classifier")
    parser.add_argument("--image-dir", type=str, default=r"datasets/CTG2.0/train/images",
                        help="Directory of images")
    parser.add_argument("--txt-dir", type=str, required=True,
                        help="Directory of label files with color flag (REQUIRED)")
    parser.add_argument("--model-path", type=str, default=r"tools/color_classifier/color_classifier_resnet18.pth",
                        help="Path to color classifier weight")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cm-output-path", type=str, default="color_classifier_confusion_matrix.png",
                        help="Save path for confusion matrix figure")
    parser.add_argument("--error-sample-dir", type=str, default="misclassified_samples",
                        help="Folder to store misclassified images")
    args = parser.parse_args()
    main(args)
