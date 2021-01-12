import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from basicblock import BasicBlock
from torch.nn import functional as F
import torch.nn.init as init

def conv3X3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class HourglassBlock(pl.LightningModule):
    '''
        define a basic block for hourglass neetwork
            ^-------------------------upper conv-------------------
            |                                                      |
            |                                                      V
        input------>downsample-->low1-->middle-->low2-->upsample-->+-->output
        NOTE about output:
            Since we need the lighting from the inner most layer,
            let's also output the results from middel layer
    '''

    def __init__(self, inplane, mid_plane, middleNet, skipLayer=True):
        super(HourglassBlock, self).__init__()
        # upper branch
        self.skipLayer = True
        self.upper = BasicBlock(inplane, inplane, batchNorm_type=1)

        # lower branch
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.low1 = BasicBlock(inplane, mid_plane)
        self.middle = middleNet
        self.low2 = BasicBlock(mid_plane, inplane, batchNorm_type=1)

    def forward(self, x, light, count, skip_count):
        # we use count to indicate wich layer we are in
        # max_count indicates the from which layer, we would use skip connections
        # print("\HourglassBlock inside forward", x.shape, light.shape)
        out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        out_lower, out_middle = self.middle(out_lower, light, count + 1, skip_count)
        out_lower = self.low2(out_lower)
        out_lower = self.upSample(out_lower)

        if count >= skip_count and self.skipLayer:
            # withSkip is true, then we use skip layer
            # easy for analysis
            out = out_lower + out_upper
        else:
            out = out_lower
            # out = out_upper
        return out, out_middle

