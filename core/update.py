import torch
import torch.functional as F
import torch.nn as nn
from torch.nn import functional as F


class preconv(nn.Module):

    def __init__(self, in_ch=3, out_ch=29):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding = 3, stride =1 ),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=2, num_channels=out_ch),
        )
        self.relu = nn.PReLU()


    def forward(self, x):
        ip = x.clone()
        feat = self.net1(x)
        y1 = torch.cat([ip, feat], dim=1)
        y1 = self.relu(y1)
        return y1


class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """

    def __init__(self, in_ch, out_ch, Pad = 1, Stride = 1):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=Pad, stride = Stride),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=2, num_channels=out_ch),
            nn.PReLU())
        self.net2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding = 1, stride = 1),
                                  nn.InstanceNorm2d(out_ch),
                                  # nn.GroupNorm(num_groups=2, num_channels=out_ch),
                                  nn.PReLU(),
                                  )

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        return x1, x2


class TripleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
            nn.PReLU())
        self.net2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding =1, stride=1),
                                  nn.InstanceNorm2d(out_ch),
                                  # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
                                  nn.PReLU(),
                                  )
        self.net3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding = 1, stride=1),
                                  nn.InstanceNorm2d(out_ch),
                                  # nn.GroupNorm(num_groups=out_ch //2, num_channels=out_ch),
                                  nn.PReLU(),
                                  )

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        return x1, x2, x3


class Down(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(self, in_ch, out_ch, Pad, Stride):
        super().__init__()
        self.net = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch, Pad , Stride)
        )

    def forward(self, x):
        return self.net(x)


# class Up(nn.Module):
#     """
#     Upsampling (by either bilinear interpolation or transpose convolutions)
#     followed by concatenation of feature map from contracting path,
#     followed by double 3x3 convolution.
#     """
#
#     def __init__(self, in_ch, out_ch, bilinear=False):
#         super().__init__()
#         self.upsample = None
#         if bilinear:
#             self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_ch, out_ch)
#
#     def forward(self, x1, x2):
#         x1 = self.upsample(x1)
#         # # Pad x1 to the size of x2
#         # diff_h = x2.shape[2] - x1.shape[2]
#         # diff_w = x2.shape[3] - x1.shape[3]
#         # x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
#         # # Concatenate along the channels axis
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
