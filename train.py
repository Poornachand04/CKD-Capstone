import torch  # Main deep learning library
import torch.nn as nn  # Neural network layers and loss functions
from torchvision import datasets, transforms, models  # Dataset + preprocessing + pretrained models
from torch.utils.data import DataLoader  # Loads data in batches
from tqdm import tqdm  # Progress bar

# ==================== CONFIGURATION ====================
DATASET = r'D:\Poorna\ckd - Copy\data\dataset_B\Kidney_Stone_Dataset'  # Dataset path
BATCH_SIZE = 32   # Number of images processed at once
EPOCHS = 10       # Number of times model sees full dataset
LR = 0.001        # Learning rate (step size for learning)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

print(f"Using device: {DEVICE}")

# ==================== IMAGE PREPROCESSING ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image (required for ResNet/DenseNet)
    transforms.ToTensor(),          # Convert image to numerical tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet values
                         [0.229, 0.224, 0.225])
])

# ==================== LOAD DATASET ====================
train_data = datasets.ImageFolder(f'{DATASET}/train', transform=transform)  # Training data
val_data = datasets.ImageFolder(f'{DATASET}/val', transform=transform)      # Validation data

# Load data in batches
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Number of classes (Normal, Cyst, Stone, Tumor)
num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ==================== TRAINING FUNCTION ====================
def train_model(model, name):
    model = model.to(DEVICE)  # Move model to GPU/CPU

    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Optimizer

    best_acc = 0  # Track best validation accuracy

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode

        correct = 0
        total = 0

        loop = tqdm(train_loader)  # Progress bar

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()  # Reset gradients

            outputs = model(images)  # Forward pass (prediction)
            loss = criterion(outputs, labels)  # Calculate loss

            loss.backward()  # Backpropagation (learn from error)
            optimizer.step()  # Update weights

            # Get predicted class
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Show training progress
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)

        # ================= VALIDATION =================
        model.eval()  # Set model to evaluation mode

        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient (faster)
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{name}_best.pth")
            print("✅ Model Saved!")

# ==================== MODEL 1: RESNET50 ====================
resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Modify final layer
train_model(resnet, "resnet50")

# ==================== MODEL 2: DENSENET121 ====================
densenet = models.densenet121(pretrained=True)  # Load pretrained DenseNet
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)  # Modify output
train_model(densenet, "densenet121")