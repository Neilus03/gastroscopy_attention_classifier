
import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()

        # Load pre-trained EfficientNetV2 model
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, dropout=0.0, stochastic_depth=0.0)

        # Replace the last layer with a new, untrained one
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),  # Batch normalization
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),  # Batch normalization
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),    # Dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),    # Dropout for regularization
            nn.Linear(64, num_classes) # Output layer
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def initialize_model(num_classes):
    model = CustomEfficientNet(num_classes)
    return model
