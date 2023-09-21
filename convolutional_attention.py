import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConvolutionalAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        
        attention_scores = F.softmax(torch.matmul(Q.view(Q.size(0), Q.size(1), -1), K.view(K.size(0), K.size(1), -1).permute(0, 2, 1)), dim=-1)
        attention_output = torch.matmul(attention_scores, V.view(V.size(0), V.size(1), -1))
        return attention_output.view_as(x)

class LinearAttention(nn.Module):
    def __init__(self, in_dim):
        super(LinearAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.t())
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

class CombinedAttentionModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CombinedAttentionModel, self).__init__()
        
        # Assuming input is a square image of size 224 for example purposes
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_attention = ConvolutionalAttention(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_attention = LinearAttention(64)
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_attention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_attention(x)
        x = self.fc(x)
        return x
