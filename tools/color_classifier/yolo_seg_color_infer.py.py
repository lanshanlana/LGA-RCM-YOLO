# yolo_seg_color_infer.py
import argparse
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
import torchvision.models as models
from torch import nn


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo_model = YOLO(args.yolo_model_path)

    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load(args.color_model_path, map_location=device))
    resnet.eval()
    resnet = resnet.to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = yolo_model(args.image_path)
    result = results[0]
    origin_image = cv2.imread(args.image_path)
    class_name_map = result.names.copy()

    if result.masks is not None:
        for class_id, mask_data in zip(result.boxes.cls, result.masks.data):
            label = result.names[int(class_id)]
            if label not in ["gas_liquid_interface", "liquid_liquid_interface"]:
                continue

            mask_np = mask_data.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))

            y_coords, x_coords = np.where(mask_resized > 0)
            if len(x_coords) == 0:
                continue
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            crop_region = origin_image[y_min:y_max, x_min:x_max]
            crop_mask = mask_resized[y_min:y_max, x_min:x_max]
            masked_crop = cv2.bitwise_and(crop_region, crop_region, mask=crop_mask.astype('uint8') * 255)

            crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = resnet(input_tensor)
                pred_class = pred.argmax(dim=1).item()
                color_tag = "colored" if pred_class == 1 else "colorless"

            class_name_map[int(class_id)] = f"{label}-{color_tag}"

    result.names = class_name_map
    output_image = result.plot()
    cv2.imwrite(args.output_path, output_image)
    print(f"Visualization saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11-Seg Combined with ResNet Color Classifier Inference")
    parser.add_argument("--yolo-model-path", type=str, default=r"assets/results/weights/best.pt",
                        help="Path of YOLO segmentation weight")
    parser.add_argument("--color-model-path", type=str, default=r"tools/color_classifier/color_classifier_resnet18.pth",
                        help="Path of color classifier weight")
    parser.add_argument("--image-path", type=str, default=r"samples/test.jpg",
                        help="Input image path for inference")
    parser.add_argument("--output-path", type=str, default=r"yolo_seg_with_color_classifier.png",
                        help="Save path of visualized result")
    args = parser.parse_args()
    main(args)
