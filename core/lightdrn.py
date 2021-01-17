import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import math
from .frn import FilterResponseNorm2d
from .convgru import ConvGRU


class ResnetDilatedBlock(pl.LightningModule):
    def __init__(self, dim, dilation_size=1, padding_size=1, normalization=True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, dilation_size, padding_size, normalization)

    def build_conv_block(self, dim, dilation_size, padding_size, normalization):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation_size, dilation=dilation_size)]
        if normalization is True:
            conv_block += [FilterResponseNorm2d(dim)]
        conv_block += [nn.LeakyReLU(0.2, True)]  # ReLU
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation_size, dilation=dilation_size)]
        if normalization is True:
            conv_block += [FilterResponseNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetDilateInjectGenerator(pl.LightningModule):
    """
    Dilated ResNet [https://arxiv.org/abs/1705.09914].
    Example usage:
        G = ResnetDilateGenerator(3, 3, (4,4,4,4))
    """
    def __init__(self, in_nc, out_nc, dilations, ngf=64):
        """
        """
        super().__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.ngf = ngf

        # preserve the scale
        self.block1 = nn.Sequential(nn.Conv2d(in_nc, ngf, kernel_size=7, padding=3),
                                    nn.LeakyReLU(0.2, True))  # ReLU

        # downsampling
        self.block2 = nn.Sequential(nn.Conv2d(ngf + 32, ngf * 2, kernel_size=3, stride=2, padding=1),
                                    FilterResponseNorm2d(ngf * 2),
                                    nn.LeakyReLU(0.2, True))  # ReLU

        # preserve the scale
        self.inner_blocks = nn.ModuleList()
        sz = 2*ngf  + 32
        for _, dilation in enumerate(dilations):
            self.inner_blocks.append(nn.Sequential(ResnetDilatedBlock(sz, dilation_size=dilation, normalization=False)))
            sz += 32

        # upsampling
        self.block3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(1), nn.Conv2d(200, int(ngf * 2 / 2), kernel_size=3, stride=1, padding=0),
                                    FilterResponseNorm2d(ngf),
                                    nn.ELU(True))

        # degrading, preserving scale
        self.block4 = nn.Sequential(nn.Conv2d(ngf + 32, ngf, kernel_size=3, dilation=1, padding=1),
                                    #FilterResponseNorm2d(ngf),
                                    nn.ELU(True))

        # generate final output image
        self.block5 = nn.Sequential(nn.Conv2d(ngf + 32 + 1, 2, kernel_size=3, padding=1),
                                    nn.ELU(True),
                                    nn.Flatten(),
                                    nn.Linear(16*32*2, 16*32*3))


    def forward(self, input_, side_channels):
        x = self.block1(input_)
        x = torch.cat((x, F.interpolate(side_channels[0][:,-32:,:,:], (16, 32), mode='nearest')), dim=1)
        x = self.block2(x)
        for i in range(len(self.inner_blocks)):
            x = torch.cat((x, F.interpolate(side_channels[i + 1][:,-32:,:,:], (8, 16), mode='nearest')), dim=1)
            x = self.inner_blocks[i](x)
        x = torch.cat((x, F.interpolate(side_channels[3][:,-32:,:,:], (8, 16), mode='nearest')), dim=1)
        x = self.block3(x)
        x = torch.cat((x, F.interpolate(side_channels[4][:,-32:,:,:], (16, 32), mode='nearest')), dim=1)
        x = self.block4(x)
        x = torch.cat((x,
                       torch.ones((1,1,16,32), dtype=x.dtype, device=x.device),
                       F.interpolate(side_channels[5][:,-32:,:,:], (16, 32), mode='nearest')), dim=1)
        x = self.block5(x)
        return x.view(-1,3,16,32)
