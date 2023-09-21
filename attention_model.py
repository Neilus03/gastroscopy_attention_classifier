import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Query, Key, Value linear layers
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        # Scaling factor
        self.scale = torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))

    def forward(self, x):
        # Calculate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores (QK^T)
        attention_scores = torch.matmul(Q, K.t()) / self.scale
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Weighted values
        weighted_values = torch.matmul(attention_weights, V)
        
        return weighted_values


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()

        # Load pre-trained EfficientNetV2 model
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, dropout=0.0, stochastic_depth=0.0)
        
        # Extract out_features from the pre-trained model
        backbone_out_features = self.backbone.classifier[0].in_features

        # Self Attention layer
        self.attention = SelfAttention(in_dim=backbone_out_features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

def initialize_model(num_classes):
    model = CustomEfficientNet(num_classes)
    return model
