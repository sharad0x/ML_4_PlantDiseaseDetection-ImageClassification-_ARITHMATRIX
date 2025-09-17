import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import os

# Load Model Function
@st.cache_resource
def load_model(num_classes, device):
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load("saved_models/plant_disease_efficientnet_b3.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

# Main App
def main():
    st.title("Plant Disease Classification App")
    st.write("Upload a leaf image and get the predicted disease category.")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    train_dataset = datasets.ImageFolder("data/train")
    classes = train_dataset.classes
    num_classes = len(classes)

    # Load model
    model = load_model(num_classes, device)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # File uploader
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = classes[pred.item()]

        # Display result
        st.success(f"✅ Predicted Class: **{predicted_class}**")

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width=300)
        with col2:
            st.markdown("### 🧾 Prediction")
            st.write(f"**Class:** {predicted_class}")

if __name__ == "__main__":
    main()
