import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


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
        self.net1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.GroupNorm(num_groups=256, num_channels=512),
            nn.InstanceNorm2d(512),
            nn.PReLU())

        self.net2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.GroupNorm(num_groups=256, num_channels=512),
            nn.InstanceNorm2d(512),
            nn.PReLU())

        self.block1 = nn.Conv2d(self.ncInput, 1536, kernel_size=(1, 1))
        self.block2 = nn.Conv2d(self.ncInput, 512, kernel_size=(1, 1))
        # self.block1 = nn.Sequential(nn.Conv2d(self.ncInput,1536, kernel_size=(1,1)),   nn.InstanceNorm2d(1535),nn.PReLU())
        # self.block2 = nn.Sequential(nn.Conv2d(self.ncInput, 512, kernel_size=(1,1)),   nn.InstanceNorm2d(512), nn.PReLU())
        self.net3 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1),
            # nn.GroupNorm(num_groups=256, num_channels=512),
            nn.InstanceNorm2d(512),
            nn.PReLU())
        self.net4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            # nn.GroupNorm(num_groups=256, num_channels=512),
            nn.InstanceNorm2d(256),
            nn.PReLU())
        self.softplus = nn.Softplus()

    def forward(self, innerFeat, targetLight):
        batch_size, ch, h, w = innerFeat.shape
        innerFeat = self.net1(innerFeat)
        innerFeat = self.net2(innerFeat)
        ip1 = innerFeat.clone()
        ip2 = innerFeat.clone()
        ip1 = self.block1(ip1)
        ip2 = self.block2(ip2)
        ip1 = ip1.unsqueeze(2)
        ip1 = ip1.unsqueeze(3)
        rgb = ip1.reshape([batch_size, 16, 32, 3, 16, 16])
        ip2 = ip2.unsqueeze(2)
        ip2 = ip2.unsqueeze(3)
        confidence = ip2.reshape([batch_size, 16, 32, 1, 16, 16])
        confidence = confidence.repeat_interleave(repeats=3, dim=3)
        confidence = confidence.reshape([-1, 16 * 16])
        confidence = self.softplus(confidence)
        # confidence = F.softmax(confidence, dim = 1)
        confidence = confidence.reshape([batch_size, 16, 32, 3, 256])
        rgb = rgb.reshape([batch_size, 16, 32, 3, 16 * 16])

        lightmap = torch.sum(rgb * confidence, dim=-1) / torch.sum(confidence)
        lightmap = lightmap.reshape([batch_size, 3, 16, 32])

        #Target light
        t1 = targetLight.reshape([batch_size, 1536, 1, 1])
        t2 = t1.repeat_interleave(repeats=16, dim=2)
        t2 = t2.repeat_interleave(repeats=16, dim=3)
        t3 = self.net3(t2)
        t3 = self.net4(t3)
        return t3, lightmap