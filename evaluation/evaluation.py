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
wandb.init(project='gastroscopy_attention_classifier_training', entity='neildlf')

# Initialize dataset
transform = transforms.Compose([
    #transforms.Resize((300, 340)),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = EGDDataset(
    image_folder="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference/all_images_mean_cropped",
    label_file="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference/full_well.xlsx",
    transform=transform)

# Initialize DataLoader with batch_size
batch_size = 16

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and load the weights
num_classes = 3
model = initialize_model(num_classes)
model_weights_path = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/kfold_weights/model_fold_3.pth"
model.load_state_dict(torch.load(model_weights_path))

# Initialize the criterion
criterion = nn.CrossEntropyLoss()

# Initialize the metrics dictionary
metrics_dict = {}

# Initialize the confusion matrix
conf_mat = np.zeros((num_classes, num_classes))

# Set the model to evaluation mode
model.eval()

correct = 0
total = 0

# Iterate over the data_loader
with torch.no_grad():
    for images, individual_labels, final_label in data_loader:
        outputs = model(images)
        loss = criterion(outputs, final_label)

        _, predicted = torch.max(outputs.data, 1)
        total += final_label.size(0)
        correct += (predicted == final_label).sum().item()

        conf_mat += confusion_matrix(final_label.cpu(), predicted.cpu(), labels=range(num_classes))
    # Calculate and print per-class accuracy and recall
    class_metrics = {}
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

        class_metrics[f"class_{i}_accuracy"] = accuracy
        class_metrics[f"class_{i}_recall"] = recall

    
    sns.heatmap(conf_mat, annot=True, cmap='cool')
    plt.title("Classification confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted as")
    fig = plt.gcf()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

    global_accuracy = correct / total
    global_recall = np.trace(conf_mat) / np.sum(conf_mat)

    print(f'Global Accuracy: {global_accuracy * 100:.2f}%')
    print(f'Global Recall: {global_recall * 100:.2f}%')
    print("\n\n")
    print("---------------------------------------------")

    # Log metrics to wandb
    wandb.log({
        "val_loss": loss.item(),
        "val_accuracy": 100 * correct / total,
        "global_accuracy": global_accuracy,
        "global_recall": global_recall,
        **class_metrics
    })

    # Store metrics in dictionary
    metrics_dict = {
        "global_accuracy": global_accuracy,
        "global_recall": global_recall,
        **class_metrics
    }

# Log metrics for all folds
wandb.log(metrics_dict)

print(metrics_dict)
wandb.finish()

