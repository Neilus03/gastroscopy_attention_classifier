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
