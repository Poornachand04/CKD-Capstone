import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATASET = r'D:\Poorna\ckd - Copy\data\dataset_B\Kidney_Stone_Dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load validation data
val_data = datasets.ImageFolder(f'{DATASET}/val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32)

classes = val_data.classes  # Class names

# ================= LOAD MODEL =================
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))  # Load weights
    model = model.to(DEVICE)
    model.eval()  # Evaluation mode
    return model

# Load models
resnet = models.resnet50()
resnet.fc = nn.Linear(resnet.fc.in_features, 4)
resnet = load_model(resnet, "resnet50_best.pth")

densenet = models.densenet121()
densenet.classifier = nn.Linear(densenet.classifier.in_features, 4)
densenet = load_model(densenet, "densenet121_best.pth")

# ================= ENSEMBLE PREDICTION =================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        # Get probabilities from both models
        out1 = F.softmax(resnet(images), dim=1)
        out2 = F.softmax(densenet(images), dim=1)

        avg = (out1 + out2) / 2  # Soft voting

        _, preds = torch.max(avg, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print("Ensemble Accuracy:", acc)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(all_labels, all_preds)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=classes,
            yticklabels=classes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()