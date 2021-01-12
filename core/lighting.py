import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from torch.nn import functional as F
import torch
def conv3X3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class lightingNet(pl.LightningModule):
    '''
        define lighting network
    '''

    def __init__(self, ncInput, ncOutput, ncMiddle):  # 27,3,27
        super(lightingNet, self).__init__()
        self.ncInput = ncInput
        self.ncOutput = ncOutput
        self.ncMiddle = ncMiddle
        self.block1 = nn.Sequential(nn.Conv2d(self.ncInput,16 * 32 * 3, kernel_size=(3,3)),nn.PReLU())
        self.block1 = nn.Sequential(nn.Conv2d(self.ncInput, 16 * 32 * 1, kernel_size=(3, 3)), nn.PReLU())
        # self.block1 = nn.Sequential(nn.Conv2d(4,4, kernel_size=(1,1),stride=(1,2),output_padding=(0,1)),nn.PReLU())
        # self.block2 = nn.Sequential(nn.ConvTranspose2d(4,4, kernel_size=(3,3), stride=(1,2),output_padding=(0,1)),
        #                              nn.PReLU())

    def forward(self, innerFeat, target_light, count, skip_count):
        # print("inside lighting", innerFeat.shape, target_light.shape)
        innerFeatold = innerFeat.clone()
        innerFeat1 = innerFeat[:, 0:4, :, :]
        innerFeat = self.block1(innerFeat1)
        _, _, row, col = innerFeat.shape
        rgb = innerFeat[:,:3,:,:]
        confidence = innerFeat[:, 3:4, :, :]
        confidence = torch.reshape(confidence,)
        confidence = F.softmax(confidence)
        ip_light = rgb * confidence

        return innerFeatold, ip_light
