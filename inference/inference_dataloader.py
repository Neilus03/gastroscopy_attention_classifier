from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


class inference_Dataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.file_list = []
            
        for filename in os.listdir(root_folder):
            file_path = os.path.join(root_folder, filename)
            if not os.path.isfile(file_path):
                continue
            self.file_list.append(file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = inference_Dataset(root_folder="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/inference_unlabelled_dataset/all_imgs_inference", 
                              transform=transform)
    
    image_tensor = dataset[42]
    image_array = np.transpose(image_tensor.numpy(), (1, 2, 0)) 
    plt.imshow(image_array)
    plt.show()
