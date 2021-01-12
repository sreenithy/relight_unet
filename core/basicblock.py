
import torch.nn as nn
import pytorch_lightning as pl

def conv3X3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# define the network
class BasicBlock(pl.LightningModule):
    def __init__(self, inplanes, outplanes, batchNorm_type=0, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # batchNorm_type 0 means batchnormalization
        #                1 means instance normalization
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, 1)
        self.conv2 = conv3X3(outplanes, outplanes, 1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        if batchNorm_type == 0:
            self.bn1 = nn.BatchNorm2d(outplanes)
            self.bn2 = nn.BatchNorm2d(outplanes)
        else:
            self.bn1 = nn.InstanceNorm2d(outplanes)
            self.bn2 = nn.InstanceNorm2d(outplanes)

        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.inplanes != self.outplanes:
            out += self.shortcuts(x)
        else:
            out += x

        out = self.prelu2(out)
        return out

