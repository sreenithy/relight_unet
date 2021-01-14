import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import numpy as np

def conv3X3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class lightingNet(pl.LightningModule):
    '''
        define lighting network
    '''

    def __init__(self, ncInput):
        super(lightingNet, self).__init__()
        self.ncInput = ncInput
        self.block1 = nn.Sequential(nn.Conv2d(self.ncInput,1536, kernel_size=(1,1)),nn.PReLU())
        self.block2 = nn.Sequential(nn.Conv2d(self.ncInput, 512, kernel_size=(1,1)), nn.PReLU())
        self.prelu = nn.PReLU()


    def forward(self, innerFeat, skip, count):
        innerFeatold = innerFeat.clone() #innerfeat size [batch_size, 128, 16,16]
        ip1 = innerFeat.clone()
        ip2 = innerFeat.clone()
        batch_size,ch,h,w = innerFeatold.shape
        ip1 = self.block1(ip1)
        ip2 = self.block2(ip2)
        ip1 = ip1.unsqueeze(2)
        ip1 = ip1.unsqueeze(3)
        rgb = ip1.reshape([batch_size,16,32,3, 16, 16])
        ip2 = ip2.unsqueeze(2)
        ip2 = ip2.unsqueeze(3)
        confidence = ip2.reshape([batch_size, 16,32,1, 16, 16])
        confidence = confidence.repeat_interleave(repeats=3, dim=3)
        confidence = confidence.reshape([-1, 16 * 16])
        confidence = F.softmax(confidence, dim = 1)
        confidence = confidence.reshape([batch_size,16,32,3,256])
        rgb = rgb.reshape([batch_size,16,32,3, 16 * 16])
        lightmap = torch.sum(rgb*confidence, dim = -1)/torch.sum(confidence)
        lightmap = lightmap.reshape([batch_size,3,16,32])
        lightmap =self.prelu(lightmap)
        return innerFeatold, lightmap
