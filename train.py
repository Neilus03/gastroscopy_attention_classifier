import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from efficientnet import initialize_model
from dataloader import EGDDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter()

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

# Split data into training, validation, and test sets
total_samples = len(dataset)
train_size = int(0.85 * total_samples)
val_size = total_samples - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Hyperparameters
num_epochs = 40
learning_rate = 1e-4

# Initialize the model and optimizer
num_classes = 3
model = initialize_model(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate )

# Loss function
criterion = nn.CrossEntropyLoss()

# Initialize lists to store losses and accuracies
train_losses = []
val_losses = []
val_accuracies = []

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, individual_labels, final_label) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, final_label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    conf_mat = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for images,individual_labels, final_label  in val_loader:
            outputs = model(images)
            loss = criterion(outputs, final_label)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += final_label.size(0)
            correct += (predicted == final_label).sum().item()

            conf_mat += confusion_matrix(final_label.cpu(), predicted.cpu(), labels=range(num_classes))

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(100 * correct / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100 * correct / total:.2f}%')
    print(f"correctly classified: {correct}, incorrectly classified: {total-correct}")
    sns.heatmap(conf_mat, annot=True,cmap='cool')
    plt.title("Classification confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted as")
    plt.show()

    # Calculate and print per-class accuracy and recall
    for i in range(num_classes):
        tp = conf_mat[i, i]
        fn = np.sum(conf_mat[i, :]) - tp
        fp = np.sum(conf_mat[:, i]) - tp
        tn = np.sum(conf_mat) - tp - fn - fp

        accuracy = (tp + tn) / np.sum(conf_mat)
        recall = tp / (tp + fn)

        print(f'Class {i} Accuracy: {accuracy * 100:.2f}%')
        print(f'Class {i} Recall: {recall * 100:.2f}%')
        print("\n")
    print("\n\n")
    print("---------------------------------------------")

