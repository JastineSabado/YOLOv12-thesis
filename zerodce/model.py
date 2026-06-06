import torch
import torch.nn as nn

class ZeroDCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3,  32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)
        self.tanh  = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        A  = self.tanh (self.conv7(torch.cat([x1, x6], dim=1)))

        enhanced = x
        for r in torch.split(A, 3, dim=1):
            enhanced = enhanced + r * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, A