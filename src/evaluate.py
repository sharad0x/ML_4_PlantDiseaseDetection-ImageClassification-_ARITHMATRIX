import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import EfficientNet_B3_Weights
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_model(num_classes, device):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = models.efficientnet_b3(weights=weights)
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

    # Data transform
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset for evaluation
    test_dataset = datasets.ImageFolder("data/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Always get class labels from TRAIN dataset (must match checkpoint)
    train_dataset = datasets.ImageFolder("data/train", transform=transform)
    classes = train_dataset.classes
    num_classes = len(classes)


    # Load model
    model = load_model(num_classes, device)

    # Ensure results folder
    os.makedirs("results", exist_ok=True)

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)
    print(report)

    # Accuracy curve (if log file exists from training)
    if os.path.exists("results/training_log.npz"):
        log = np.load("results/training_log.npz")
        train_acc = log["train_acc"]
        val_acc = log["val_acc"]
        train_loss = log["train_loss"]
        val_loss = log["val_loss"]

        # Accuracy curve
        plt.plot(train_acc, label="Train Acc")
        plt.plot(val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")
        plt.savefig("results/accuracy_curve.png")
        plt.close()

        # Loss curve
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig("results/loss_curve.png")
        plt.close()

        print("Saved accuracy_curve.png and loss_curve.png")
    else:
        print("No training_log.npz found, skipping curves.")

if __name__ == "__main__":
    main()
