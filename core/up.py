import torch
import torch.functional as F
import torch.nn as nn
from torch.nn import functional as F

class TripleUp(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.net1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
            nn.PReLU())
        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
            nn.PReLU())
        self.net3 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            # nn.GroupNorm(num_groups=out_ch // 2, num_channels=256),
            nn.PReLU())

    def forward(self, x1, x40, x41, x42):
        # x1 = self.upsample(x1)
        y1 = torch.cat([x1, x42], dim=1)
        y2 = self.net1(y1)
        y2 = torch.cat([y2, x41], dim=1)
        y2 = self.net2(y2)
        y2 = torch.cat([y2, x40], dim=1)
        y2 = self.net3(y2)
        y2 = F.pad(y2, [1 // 2, 1 - 1 // 2, 1 // 2, 1 - 1 // 2])
        return y2

class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
            nn.PReLU())

        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch // 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch // 2),
            # nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.PReLU())

    def forward(self, x, ip0, ip1):

        x = torch.cat([x, ip1], dim=1)
        x = self.net1(x)
        x = torch.cat([x, ip0], dim=1)
        x = self.net2(x)
        x = F.pad(x, [1 // 2, 1 - 1 // 2, 1 // 2, 1 - 1 // 2])
        return x
