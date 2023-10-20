import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from efficientnet import initialize_model
from inference_dataloader import inference_Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from PIL import Image
import cv2
import wandb


transform = transforms.Compose([
    transforms.Resize((300, 340)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = inference_Dataset(
    root_folder="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference/testset_label0",
    transform=transform,
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model
num_classes = 3
model = initialize_model(num_classes)

# Load averaged model weights (THEY DON´T SEEM TO WORK WELL, SPECIALLY WRONG FOR CLASS 1 CLASSIFICATION)
avg_model_weights_path = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference/best_model_fold_2.pth"
model.load_state_dict(torch.load(avg_model_weights_path))

# Evaluate the model
model.eval()

wandb.init(project="inference_egd", entity="neildlf")
with torch.no_grad():
    for images in val_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        print("label:",predicted.item())
        wandb.log({"predicted_label": predicted[0]})
        plt.imshow(images[0].permute(1, 2, 0))
        wandb.log({"image": wandb.Image(images[0])})
        plt.title(f"Predicted label: {predicted[0]}")
        plt.show()
        
wandb.finish()