import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from efficientnet import initialize_model  
from dataloader import EGDDataset  
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter()

# Initialize dataset
transform = transforms.Compose([
    transforms.Resize((300,340)), #originally 560 x 640 so we preserve the aspect ratio
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = EGDDataset(
    image_folder="C:/Users/neild/OneDrive/Documentos/CVC/EGD_Barcelona/gastroscopy_attention_classifier/all_images_mean_cropped",
    label_file="C:/Users/neild/OneDrive/Documentos/CVC/EGD_Barcelona/gastroscopy_attention_classifier/full_filtered.xlsx",
    transform=transform
)

# Split data into training, validation, and test sets
total_samples = len(dataset)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Function to perform training
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        running_loss = 0.0
        for i, (images, individual_labels, final_label) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, final_label)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Write to TensorBoard
            if i % 10 == 9:  # Log every 10 batches
                writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0

        # Validation loss
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, individual_labels, final_label in val_loader:
                outputs = model(images)
                loss = criterion(outputs, final_label)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += final_label.size(0)
                correct += (predicted == final_label).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Write validation loss and accuracy to TensorBoard
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Initialize the model and optimizer
num_classes = 3  # As determined before
model = initialize_model(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function
criterion = nn.CrossEntropyLoss()

# Train the model
train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer)

# Close the TensorBoard writer
writer.close()
