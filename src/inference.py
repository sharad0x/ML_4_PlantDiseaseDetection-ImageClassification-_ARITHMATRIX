import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import os
import matplotlib.pyplot as plt

def load_model(num_classes, device):
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load("saved_models/plant_disease_efficientnet_b3.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load classes
    train_dataset = datasets.ImageFolder("data/train")
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes")

    # Load model
    model = load_model(num_classes, device)

    # Inference folder
    inference_folder = "data/inference"
    image_files = [f for f in os.listdir(inference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) == 0:
        print("No images found in data/inference/")
        return

    # Limit to first 12 for grid (adjust as needed)
    image_files = image_files[:12]

    plt.figure(figsize=(16, 12))

    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(inference_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = classes[pred.item()]

        # Plot in grid
        plt.subplot(3, 4, idx)  # 3x4 grid
        plt.imshow(image)
        plt.title(f"{predicted_class}", fontsize=10)
        plt.axis("off")

        print(f"{img_file} → {predicted_class}")

    plt.tight_layout()
    os.makedirs("results/inference", exist_ok=True)
    save_path = "results/inference/inference_grid.png"
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved grid to {save_path}")

if __name__ == "__main__":
    main()
