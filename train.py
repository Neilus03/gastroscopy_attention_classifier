import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from efficientnet import initialize_model
from dataloader import EGDDataset
from train_setup import *
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize k-Fold cross-validation
k_folds = 5 
kfold = KFold(n_splits=k_folds, shuffle=True)

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

