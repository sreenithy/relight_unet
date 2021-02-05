import torch
import torch.functional as F
import torch.nn as nn
from torch.nn import functional as F

class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.PReLU(),
        )
        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch , kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch ),
            # nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.PReLU())

    def forward(self, x1, x20,x21):
        x1 = self.upsample1(x1)
        # Pad x1 to the size of x2
        diff_h = x21.shape[2] - x1.shape[2]
        diff_w = x21.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        # Concatenate along the channels axis
        x = torch.cat([x21, x1], dim=1)
        x = self.net1(x)
        x = torch.cat([x20, x], dim=1)
        x = self.net2(x)
        return x


class TripleUp(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.net1 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=512 // 2, num_channels=512),
            nn.PReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=in_ch // 4, num_channels=in_ch // 2),
            nn.PReLU())
        self.net3 = nn.Sequential(
            nn.Conv2d(in_ch, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            # nn.GroupNorm(num_groups=out_ch // 2, num_channels=512),
            nn.PReLU())

    def forward(self, x1, x20, x21, x23):
        x = torch.cat([x23, x1], dim=1)
        x =  self.net1(x)
        x = torch.cat([x21, x], dim=1)
        x = self.net2(x)
        x = torch.cat([x20, x], dim=1)
        x = self.net3(x)
        return x
