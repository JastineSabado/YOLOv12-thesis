import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorConstancyLoss(nn.Module):
    def forward(self, x):
        r, g, b = x[:,0], x[:,1], x[:,2]
        return (torch.mean(r-g)**2 +
                torch.mean(g-b)**2 +
                torch.mean(b-r)**2) / 3.0

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernels = [
            torch.tensor([[0,0,0],[-1,1,0],[0,0,0]], dtype=torch.float32),
            torch.tensor([[0,0,0],[0,1,-1],[0,0,0]], dtype=torch.float32),
            torch.tensor([[0,-1,0],[0,1,0],[0,0,0]], dtype=torch.float32),
            torch.tensor([[0,0,0],[0,1,0],[0,-1,0]], dtype=torch.float32),
        ]
    def forward(self, enhanced, original):
        loss = 0
        for k in self.kernels:
            kernel = k.unsqueeze(0).unsqueeze(0).to(enhanced.device).expand(3,1,3,3)
            loss += torch.mean(
                (F.conv2d(enhanced, kernel, padding=1, groups=3) -
                 F.conv2d(original, kernel, padding=1, groups=3)) ** 2)
        return loss

class ExposureLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x):
        gray = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).unsqueeze(1)
        return torch.mean((self.pool(gray) - self.mean_val) ** 2)

class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, A):
        m = torch.mean(A, dim=1, keepdim=True)
        return (torch.mean(torch.abs(m[:,:,:,:-1] - m[:,:,:,1:])) +
                torch.mean(torch.abs(m[:,:,:-1,:] - m[:,:,1:,:])))

class ZeroDCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.color      = ColorConstancyLoss()
        self.spatial    = SpatialConsistencyLoss()
        self.exposure   = ExposureLoss()
        self.smoothness = IlluminationSmoothnessLoss()
    def forward(self, enhanced, original, A):
        lc  = self.color(enhanced)
        ls  = self.spatial(enhanced, original)
        le  = self.exposure(enhanced)
        lsm = self.smoothness(A)
        total = ls*1.0 + le*10.0 + lc*5.0 + lsm*200.0
        return total, ls, le, lc, lsm