"""
Run inference on foreign (out-of-distribution) inputs.

Foreign inputs are images from the RPS test set that were NOT used during training.
Expected files in foreign_inputs/:
  - rock_example.jpg
  - paper_example.jpg
  - scissors_example.jpg
  - ambiguous_gesture.jpg  (extra example)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

MODEL_PATH = Path("model.pt")
FOREIGN_DIR = Path("foreign_inputs")


class RPSClassifier(nn.Module):
    """Must match architecture in train.py."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 128)

    model = RPSClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, transform


def predict_image(model, image_path: Path, class_names, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    prediction = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return prediction, confidence, prob_dict


def main():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found at {MODEL_PATH}")
        print("Run train.py first to create the model.")
        sys.exit(1)

    if not FOREIGN_DIR.exists():
        print(f"Error: foreign inputs directory not found at {FOREIGN_DIR}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, transform = load_model(MODEL_PATH, device)

    image_files = sorted(
        p
        for p in FOREIGN_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not image_files:
        print(f"No image files found in {FOREIGN_DIR}")
        sys.exit(1)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Classes: {class_names}")
    print(f"\nPredicting on {len(image_files)} foreign input(s):\n")
    print("-" * 60)

    for img_path in image_files:
        prediction, confidence, probs = predict_image(
            model, img_path, class_names, transform, device
        )
        print(f"Image: {img_path.name}")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Probabilities: {', '.join(f'{k}={v:.2%}' for k, v in probs.items())}")
        print("-" * 60)


if __name__ == "__main__":
    main()
