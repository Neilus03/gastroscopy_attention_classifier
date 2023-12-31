from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
from sklearn.model_selection import KFold

class EGDDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.labels = pd.read_excel(label_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image loading
        img_name = os.path.join(self.image_folder, f"{idx+1}.BMP") 
        image = Image.open(img_name)

        # Labels
        individual_labels = torch.tensor(self.labels.iloc[idx, :-1].values, dtype=torch.long)  # All but the last column (INDIVIDUAL DOCTORS)
        final_label = torch.tensor(self.labels.iloc[idx, -1], dtype=torch.long)  # The last column (FINAL)

        if self.transform:
            image = self.transform(image)

        return image, individual_labels, final_label

# Example usage
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = EGDDataset(image_folder="/content/drive/MyDrive/EGD-Barcelona/merged/all_images_mean_cropped", label_file="/content/drive/MyDrive/EGD-Barcelona/merged/full_well.xlsx", transform=transform)
    print(dataset[2])  # Output will be (image, individual_labels, final_label)
