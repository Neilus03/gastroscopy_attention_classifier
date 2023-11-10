import wandb
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

# Initialize wandb
wandb.init(project='gastroscopy_classifier_training_0vs12', entity='neildlf')

# Initialize dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])



root_dir = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/training/0vs12/0-12"

#set cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = EGDDataset(
    root_dir=root_dir,
    transform=transform,
    use_test_data=True  # Set to True to include test data in K-fold
)

# Initialize KFold
n_splits = 4
kfold = KFold(n_splits=n_splits, shuffle=True)

# Hyperparameters
num_epochs = 230
learning_rate = 5e-4
num_classes = 2
batch_size = 12

# Initialize sum of state_dicts to zero, for calculating mean later
sum_state_dict = None

# Store average confusion matrices for each fold and the global average
confusion_matrices = []
fold1_confusion_matrices = []
fold2_confusion_matrices = []
fold3_confusion_matrices = []
fold4_confusion_matrices = []

# Initialize dictionary to store metrics for each fold
metrics_dict = {}

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
    model = initialize_model(num_classes).to(device)
    '''
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.size()) > 1:
                nn.init.xavier_uniform_(param)
    '''       
    '''If i wanted to initalize the model with the pretrained way, uncomment next 2 lines'''
    #model_weights_path = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/kfold_weights/model_fold_3.pth"
    #model.load_state_dict(torch.load(model_weights_path))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []
    
    # Initialize confusion matrix for current fold
    fold_conf_mat = np.zeros((num_classes, num_classes))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0


        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        conf_mat = np.zeros((num_classes, num_classes))
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                conf_mat += confusion_matrix(labels.cpu(), predicted.cpu(), labels=range(num_classes))
                torch.cuda.empty_cache()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(100 * correct / total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/kfold_weights_k4_0vs12/model_fold_{fold}.pth")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {100 * correct / total:.2f}%")
        print(f"correctly classified: {correct}, incorrectly classified: {total-correct}")

        # Store the confusion matrix for the current fold
        if epoch > 100: #to do average when already trained, not when starting training
            confusion_matrices.append(conf_mat)
        if fold == 0 and epoch > 100: #to do average when already trained, not when starting training
            fold1_confusion_matrices.append(conf_mat)
        elif fold == 1 and epoch > 100: #to do average when already trained, not when starting training
            fold2_confusion_matrices.append(conf_mat)
        elif fold == 2 and epoch > 100: #to do average when already trained, not when starting training
            fold3_confusion_matrices.append(conf_mat)
        elif fold == 3 and epoch > 100: #to do average when already trained, not when starting training
            fold4_confusion_matrices.append(conf_mat)
            
        # Log the confusion matrix for the current fold to wandb with a unique name
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='g', cmap='cool')
        plt.title(f"Fold {fold} Confusion Matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        # You should save the plot before logging it to wandb
        plt.savefig(f'confusion_matrix_fold_{fold}.png')
        plt.close()
        # Log the confusion matrix with a unique name for each fold
        wandb.log({f"confusion_matrix_fold_{fold}": wandb.Image(f'confusion_matrix_fold_{fold}.png')})

        
        # Calculate and print per-class accuracy and recall
        class_metrics = {}
        for i in range(num_classes):
            tp = conf_mat[i, i]
            fn = np.sum(conf_mat[i, :]) - tp
            fp = np.sum(conf_mat[:, i]) - tp
            tn = np.sum(conf_mat) - tp - fn - fp

            accuracy = (tp + tn) / np.sum(conf_mat)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f'Class {i} Accuracy: {accuracy * 100:.2f}%')
            print(f'Class {i} Recall: {recall * 100:.2f}%')
            print(f'Class {i} Precision: {precision * 100:.2f}%')
            print(f'Class {i} F1 Score: {f1_score * 100:.2f}%')
            print("\n")

            class_metrics[f"class_{i}_accuracy"] = accuracy
            class_metrics[f"class_{i}_recall"] = recall
            class_metrics[f"class_{i}_precision"] = precision
            class_metrics[f"class_{i}_f1_score"] = f1_score

        global_accuracy = correct / total
        global_recall = np.trace(conf_mat) / np.sum(conf_mat)

        print(f'Global Accuracy: {global_accuracy * 100:.2f}%')
        print(f'Global Recall: {global_recall * 100:.2f}%')
        print("\n\n")
        print("---------------------------------------------")

        # Log metrics to wandb
        wandb.log({
            "train_loss": running_loss,
            "val_loss": val_loss,
            "val_accuracy": 100 * correct / total,
            "global_accuracy": global_accuracy,
            "global_recall": global_recall,
            **class_metrics
        })

        # Store metrics in dictionary
        metrics_dict[f"fold_{fold}"] = {
            "global_accuracy": global_accuracy,
            "global_recall": global_recall,
            **class_metrics
        }

    # Accumulate state_dicts for later averaging
    if sum_state_dict is None:
        sum_state_dict = model.state_dict()
    else:
        for key in sum_state_dict.keys():
            sum_state_dict[key] += model.state_dict()[key]
            
 
# Calculate the average confusion matrix
avg_conf_mat = np.mean(confusion_matrices, axis=0)

# Get the averaged confusion matrix for each fold
avg_conf_mat_fold1 = np.mean(fold1_confusion_matrices, axis=0) 
avg_conf_mat_fold2 = np.mean(fold2_confusion_matrices, axis=0)
avg_conf_mat_fold3 = np.mean(fold3_confusion_matrices, axis=0)
avg_conf_mat_fold4 = np.mean(fold4_confusion_matrices, axis=0)

# Plot the averaged confusion matrix for each fold

# Fold 1
sns.heatmap(avg_conf_mat_fold1, annot=True, fmt='.2f', cmap='cool')
plt.title("Fold 1 Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# Save the figure before displaying it
plt.savefig('fold1_averaged_conf_mat.png')
plt.close()
wandb.log({"fold1_avg_confusion_matrix": wandb.Image('fold1_averaged_conf_mat.png')})


#Fold 2 
sns.heatmap(avg_conf_mat_fold2, annot=True, fmt='.2f', cmap='cool')
plt.title("Fold 2 Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# Save the figure before displaying it
plt.savefig('fold2_averaged_conf_mat.png')
plt.close() 
wandb.log({"fold2_avg_confusion_matrix": wandb.Image('fold2_averaged_conf_mat.png')})

# Fold 3
sns.heatmap(avg_conf_mat_fold3, annot=True, fmt='.2f', cmap='cool')
plt.title("Fold 3 Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# Save the figure before displaying it
plt.savefig('fold3_averaged_conf_mat.png')
plt.close()
wandb.log({"fold3_avg_confusion_matrix": wandb.Image('fold3_averaged_conf_mat.png')})

# Fold 4
sns.heatmap(avg_conf_mat_fold4, annot=True, fmt='.2f', cmap='cool')
plt.title("Fold 4 Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# Save the figure before displaying it
plt.savefig('fold4_averaged_conf_mat.png')
plt.close()
wandb.log({"fold4_avg_confusion_matrix": wandb.Image('fold4_averaged_conf_mat.png')})

# Plot the global averaged confusion matrix
sns.heatmap(avg_conf_mat, annot=True, fmt='.2f', cmap='cool')
plt.title("Global Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# Save the figure before displaying it
plt.savefig('global averaged_conf_mat.png')

plt.close()

# Log the averaged confusion matrix to wandb
avg_conf_mat_fig = plt.figure(figsize=(10, 8))
sns.heatmap(avg_conf_mat, annot=True, fmt='.2f', cmap='cool')
plt.title("Global Averaged Classification Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.close(avg_conf_mat_fig)
wandb.log({"avg_confusion_matrix": wandb.Image(avg_conf_mat_fig)})

# Log metrics for all folds
wandb.log(metrics_dict)

print(metrics_dict)

wandb.finish()
