import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class labels
CLASS_NAMES = ['Normal', 'Cyst', 'Stone', 'Tumor']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize image
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # Normalize
])

# Load model
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# ================= MODEL PATH =================
MODEL_PATH = r"D:\Poorna\CKD-Capstone"

# ================= LOAD RESNET =================
resnet = models.resnet50()
resnet.fc = nn.Linear(resnet.fc.in_features, 4)

resnet_path = f"{MODEL_PATH}\\resnet50_best.pth"
print("Loading:", resnet_path)

resnet = load_model(resnet, resnet_path)

# ================= LOAD DENSENET =================
densenet = models.densenet121()
densenet.classifier = nn.Linear(densenet.classifier.in_features, 4)

densenet_path = f"{MODEL_PATH}\\densenet121_best.pth"
print("Loading:", densenet_path)

densenet = load_model(densenet, densenet_path)

# ================= PREDICTION FUNCTION =================
def predict(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image
    image = transform(image).unsqueeze(0).to(DEVICE)  # Preprocess

    with torch.no_grad():
        # Get outputs from both models
        out1 = F.softmax(resnet(image), dim=1)
        out2 = F.softmax(densenet(image), dim=1)

        final = (out1 + out2) / 2  # Ensemble

        probs = final[0].cpu().numpy()
        pred = probs.argmax()

    # Output result
    print("\nPrediction:", CLASS_NAMES[pred])
    print("Confidence:", probs[pred]*100)

    # Medical message
    if pred == 0:
        print("✓ Normal Kidney")
    else:
        print(f"⚠️ CKD Detected - {CLASS_NAMES[pred]}")

# Run prediction
# List of test images
test_images = [
    "test_image.png",
    "test_image2.png",
    "test_image3.png",
    "test_image4.png",
    "test_image5.png",
    "test_image6.png"
]

# Loop through all images
for img in test_images:
    print("\n==============================")
    print(f"Testing: {img}")
    predict(f"D:\Poorna\CKD-Capstone\{img}")  # Replace with your image path