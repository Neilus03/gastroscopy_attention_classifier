import torch
import torch.nn as nn
import torch.nn.functional as F
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        #squeeze operation is just a global average pooling that´s done in forward pass

        #excitation operation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        #GAP
        y = x.view(b, c, -1).mean(dim=2)
        
        y = self.fc(y)
        
        y = y.view(b,c,1,1)
        
        return y * x


seblock = SEBlock(64)
input_tensor = torch.rand(16,64,32,32) #batch size x channels x H x W
output_tensor = seblock(input_tensor)
print(output_tensor.size())   #torch.Size([16, 64, 32, 32])
