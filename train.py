from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import EGDDataset
from efficientnet import initialize_model
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize TensorBoard SummaryWriter
# writer = SummaryWriter()

# Initialize dataset
transform = transforms.Compose([
    transforms.Resize((300, 340)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = EGDDataset(
    image_folder="/content/drive/MyDrive/EGD-Barcelona/merged/all_images_mean_cropped",
    label_file="/content/drive/MyDrive/EGD-Barcelona/merged/full_well.xlsx",
    transform=transform
)

# Initialize KFold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True)

# Hyperparameters
num_epochs = 40
learning_rate = 1e-4
num_classes = 3
batch_size = 16

# K-Fold Cross Validation
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

    print(f"FOLD {fold}")
    print("--------------------------------")

    # Define train and validation data subsets
    train_subsampler = Subset(dataset, train_ids)
    val_subsampler = Subset(dataset, val_ids)

    # Initialize DataLoader with batch_size
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = initialize_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, individual_labels, final_label) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, final_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        conf_mat = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for images, individual_labels, final_label in val_loader:
                outputs = model(images)
                loss = criterion(outputs, final_label)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += final_label.size(0)
                correct += (predicted == final_label).sum().item()

                conf_mat += confusion_matrix(final_label.cpu(), predicted.cpu(), labels=range(num_classes))

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(100 * correct / total)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100 * correct / total:.2f}%")

        sns.heatmap(conf_mat, annot=True, cmap='cool')
        plt.title("Classification confusion matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted as")
        plt.show()
