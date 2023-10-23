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
    #transforms.Resize((300, 340)),
    #transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

val_dataset = inference_Dataset(
    root_folder="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference/all_imgs_inference",
    transform=transform,
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model
num_classes = 3
model = initialize_model(num_classes)

# Load averaged model weights (THEY DON´T SEEM TO WORK WELL, SPECIALLY WRONG FOR CLASS 1 CLASSIFICATION)
avg_model_weights_path = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/kfold_weights/model_fold_4.pth"
model.load_state_dict(torch.load(avg_model_weights_path))

# Evaluate the model
model.eval()

label1 = 0
label2 = 0
label0 = 0

wandb.init(project="inference_egd", entity="neildlf")
with torch.no_grad():
    for images in val_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        print("label:",predicted.item())
        if predicted.item() == 0:
            label0 += 1
        elif predicted.item() == 1:
            label1 += 1
        elif predicted.item() == 2:
            label2 += 1
        wandb.log({"predicted_label": predicted[0]})
        plt.imshow(images[0].permute(1, 2, 0))
        wandb.log({"image": wandb.Image(images[0])})
        plt.title(f"Predicted label: {predicted[0]}")
        fig = plt.gcf()
        wandb.log({"prediction": wandb.Image(fig)})
        plt.close()
        
print("label0:",label0, "images")
print("label1:",label1, "images")
print("label2:",label2, "images")        
wandb.finish()