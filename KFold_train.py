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
num_epochs = 20
learning_rate = 1e-4
num_classes = 3
batch_size = 16

# Initialize sum of state_dicts to zero, for calculating mean later
sum_state_dict = None

# K-Fold Cross Validation
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    best_val_loss = float('inf')

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"/content/drive/MyDrive/EGD-Barcelona/merged/effnet_weights/best_model_fold_{fold}.pth")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100 * correct / total:.2f}%")
        print(f"correctly classified: {correct}, incorrectly classified: {total-correct}")

        sns.heatmap(conf_mat, annot=True, cmap='cool')
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
    
    # Accumulate state_dicts for later averaging
    if sum_state_dict is None:
        sum_state_dict = model.state_dict()
    else:
        for key in sum_state_dict.keys():
            sum_state_dict[key] += model.state_dict()[key]

# Calculate mean state_dict
for key in sum_state_dict.keys():
    sum_state_dict[key] /= n_splits

# Load mean state_dict into a fresh model
mean_model = initialize_model(num_classes)
mean_model.load_state_dict(sum_state_dict)
torch.save(mean_model.state_dict(), "mean_model.pth")
    
