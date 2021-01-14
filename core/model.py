import sys
from itertools import zip_longest
from pathlib import Path
from argparse import ArgumentParser

sys.path.append('functions')
sys.path.append('utils')
from utils.gaussian import *
from utils.spharm import generate_input_channels
from lightstage import LightStageFrames
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image
from utils.metrics import *
from functions.func import *
import os
import pandas as pd

from functions.func import *

if not os.path.exists('results_face'):
    os.makedirs('results_face')
if not os.path.exists('results_light'):
    os.makedirs('results_light')


from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from lightstage import *
from PIL import Image
from torch.nn import functional as F
from update import HourglassBlock
from convgru import ConvGRU
from lighting import lightingNet


# From https://github.com/pytorch/pytorch/issues/15849
class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



class gradientLayer(nn.Module):
    '''
    	get the gradient of x, y direction
    '''

    def __init__(self):
        super(gradientLayer, self).__init__()
        self.weight_x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
            [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]), 0), 0)
        self.weight_y = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), 0), 0)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight.data = self.weight_x
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y.weight.data = self.weight_y

    def forward(self, x):
        out_x = self.conv_x(x)
        out_y = self.conv_y(x)
        out = torch.cat((out_x, out_y), 1)
        return out


class gradientLayer_color(nn.Module):
    '''
    	get the gradient of x, y direction
	for color images
    '''

    def __init__(self):
        super(gradientLayer_color, self).__init__()
        self.weight_x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
            [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]), 0), 0)
        self.weight_y = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), 0), 0)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight.data = self.weight_x
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y.weight.data = self.weight_y

    def forward(self, x):
        # number of channel
        # channel 1
        tmpX_0 = x[:, 0, :, :].unsqueeze(dim=1)
        outx_0 = self.conv_x(tmpX_0)
        outy_0 = self.conv_y(tmpX_0)

        tmpX_1 = x[:, 1, :, :].unsqueeze(dim=1)
        outx_1 = self.conv_x(tmpX_1)
        outy_1 = self.conv_y(tmpX_1)

        tmpX_2 = x[:, 2, :, :].unsqueeze(dim=1)
        outx_2 = self.conv_x(tmpX_2)
        outy_2 = self.conv_y(tmpX_2)

        out = torch.cat((outx_0, outx_1, outx_2, outy_0, outy_1, outy_2), 1)
        return out



class HourglassNet(pl.LightningModule):

    def __init__(self, hparams):
        super(HourglassNet, self).__init__()
        # self.hparams = hparams
        self.hparams = hparams
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.log_images = hparams.log_images
        self.weight_decay = hparams.weight_decay
        self.hid_dim = hparams.hidden_dim  # Hidden dimension GRU
        self.ip_dim = hparams.ip_dim  # Hidden dimension GRU
        self.num_workers = hparams.num_workers

        self.psnrtable = pd.DataFrame()
        self.blur = GaussianBlur(5)

        self.ncLight = 4  #  number of channels for input to lighting network
        self.baseFilter = 16


        self.ncPre = self.baseFilter  # number of channels for pre-convolution

        self.ncHG4 = self.baseFilter
        self.ncHG3 = 2 * self.baseFilter
        self.ncHG2 = 4 * self.baseFilter
        self.ncHG1 = 8 * self.baseFilter
        # self.ncHG0 = 16 * self.baseFilter + self.ncLight

        self.pre_conv = nn.Conv2d(3, self.ncPre, kernel_size=5, stride=1, padding=2)
        self.pre_bn = nn.BatchNorm2d(self.ncPre)

        self.gru = ConvGRU(input_size=self.ncPre, hidden_sizes=[64, self.ncPre],
                           kernel_sizes=[3, 3], n_layers=2)
        self.light = lightingNet(128)
        # self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, self.light)
        self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.light)
        self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
        self.HG3 = HourglassBlock(self.ncHG4, self.ncHG3, self.HG2)
        self.HG4 = HourglassBlock(self.ncPre, self.ncHG4, self.HG3)

        self.conv_1 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.InstanceNorm2d(self.ncPre)
        self.conv_2 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.InstanceNorm2d(self.ncPre)#self.bn_2 = nn.BatchNorm2d(self.ncPre)
        self.conv_3 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.InstanceNorm2d(self.ncPre)#self.bn_3 = nn.BatchNorm2d(self.ncPre)

        self.output = nn.Conv2d(self.ncPre, 3, kernel_size=1, stride=1, padding=0)
        self.save_hyperparameters()


    def forward(self, x, input_state = None):
        skip_count=0
        feat = self.pre_conv(x)
        feat = F.relu(self.pre_bn(feat))
        feat, light_estim = self.HG4(feat,0,skip_count)
        gru_features = self.gru(feat, input_state)
        # print("4. Albedo Net full out", feat.shape, full_face.shape)
        feat = F.relu(self.bn_1(self.conv_1(gru_features[-1])))
        feat = F.relu(self.bn_2(self.conv_2(feat)))
        feat = F.relu(self.bn_3(self.conv_3(feat)))
        out_img = self.output(feat)
        albedo_estim = torch.sigmoid(out_img)

        return albedo_estim, light_estim, gru_features



    def training_step(self, batch, batch_nb):

        inputs, _, light_inputs, _, albedo_gts, masks = batch
        gru_state = None
        loss = 0

        for idx, (input_, light_input, mask,albedo_gt) in enumerate(zip(inputs,light_inputs,masks,albedo_gts)):

            face_estim, light_estim, gru_state = self.forward(input_, gru_state)
            # Calculate loss
            sz = albedo_gt.size(2) ** 2
            # change
            l1_face = 5 / sz * torch.sum(torch.abs(face_estim - albedo_gt) * mask) / face_estim.shape[0]
            l1_light = 1 / (16 * 32) * torch.sum(torch.abs(light_estim - light_input)) / face_estim.shape[0]

            loss += l1_face + l1_light  # + tv_loss_albedo

            # save learning_rate
            lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
            lr_saved = torch.scalar_tensor(lr_saved).to(self.HG4.device)

            if self.log_images:
                if self.global_step % 10 == 0:
                    light_estim = F.interpolate(light_estim, (128, 256), mode="bilinear")
                    light_input = F.interpolate(light_input, (128, 256), mode="bilinear")
                    albedo_diff = 3 * torch.abs(face_estim[0:1, ...] - albedo_gt[0:1, ...])
                    img_stack = torch.clamp(torch.cat((face_estim[0:1, ...], albedo_diff, albedo_gt[0:1, ...])), min=0,
                                            max=1)
                    albedo_grid = make_grid_with_labels(img_stack.detach().cpu(), ["Output", "Diff (3x)", "Target"],
                                                        nrow=3)
                    light_diff = 3 * torch.abs(light_estim[0:1, ...] - light_input[0:1, ...])
                    img_stack = torch.clamp(torch.cat((light_estim[0:1, ...], light_diff, light_input[0:1, ...])),
                                            min=0, max=1)
                    light_grid = make_grid_with_lightlabels(img_stack.detach().cpu(), ["Output", "Diff (3x)", "Target"],
                                                            nrow=3)

                    # self.logger.experiment.add_image('epoch_{}_step_{}_face_images_{}'.format(self.current_epoch, self.global_step, idx),albedo_grid, 0)

                    save_image(albedo_grid,
                               'results_face/epoch_{}_step_{}_face_images_{}.png'.format(self.current_epoch,
                                                                                         self.global_step, idx))

                    # self.logger.experiment.add_image('epoch_{}_step_{}_face_images_{}'.format(self.current_epoch, self.global_step, idx),light_grid, 0)

                    save_image(light_grid,
                               'results_light/epoch_{}_step_{}_light_images_{}.png'.format(self.current_epoch,
                                                                                           self.global_step, idx))

                    plt.close()
            if self.hparams.log_graph == 1:
                # Logging the computational graph on tensorboard
                if self.global_step == 1:
                    example_input_array = list()
                    example_input_array.append(torch.rand((1, 3, 256, 256)))
                    self.logger.experiment.add_graph(HourglassNet(self.hparams), example_input_array)
                    print("Logged computational Graph")

            if self.hparams.log_histogram == 1:
                self.custom_histogram_adder()


        loss = loss / idx
        self.log('l1_face', l1_face, prog_bar=True)
        self.log('l1_light', l1_light, prog_bar=True)
        self.log('learning_rate', lr_saved, prog_bar=True)
        self.log('train_loss', loss)

        return loss


    def validation_step(self, batch, batch_nb):
        print("Validation")
        inputs, _, light_inputs, _, albedo_gts, masks = batch
        gru_state = None
        loss = 0
        psnr_ = []
        for idx, (input_, light_input, mask,albedo_gt) in enumerate(zip(inputs,light_inputs,masks,albedo_gts)):

            face_estim, light_estim, gru_state = self.forward(input_, gru_state)

            face_estim = torch.clamp(face_estim, min=0, max=1)
            # light_estim = torch.clamp(light_estim, min=0, max=1)

            # Calculate loss
            sz = albedo_gt.size(2) ** 2
            l1_face = 5 / sz * torch.sum(torch.abs(face_estim - albedo_gt) * mask) / face_estim.shape[0]
            l1_light = 1 / (16 * 32) * torch.sum(torch.abs(light_estim - light_input)) / face_estim.shape[0]

            loss += l1_face + l1_light

            psnr_.append(psnr(image_pred=face_estim, image_gt=albedo_gt))

        loss = loss / idx
        self.log('val_loss', loss)
        self.log('val_loss', loss, prog_bar=True)

        return {
            "val_loss": loss,
            "val_psnr": psnr_
        }


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.psnr_table(outputs, 'val_psnr')
        self.log('avg_valloss', avg_loss, prog_bar=True)
        return {'avg_valloss': avg_loss}

    def psnr_table(self, outputs, tag):
        psnr = []
        for i in range(len(outputs)):
            psnr.append(outputs[i][tag])
        avg_psnr = np.nanmean(np.array(list(zip_longest(*psnr)), dtype=float), axis=1)
        avg_psnr = avg_psnr.reshape(1, len(avg_psnr))
        avg_psnr = pd.DataFrame(avg_psnr)
        filename = self.logger.root_dir + '/version_' + str(self.logger.version-1 ) + '/' + tag + '.csv'
        avg_psnr.to_csv(filename, mode='a', sep=' ', header=False, )

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     betas=(0.5, 0.999))
        scheduler =  MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.5)
        return [optimizer], [scheduler]

    def __dataloader(self):
        dataset_train = LightStageFrames(Path("train/"))
        dataset_val = LightStageFrames(Path("val/"))
        train_loader = FastDataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        return {'train': train_loader,'val':val_loader}

    # @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--log_images', default=1, type=int,
                            help='Log intermediate training results')
        parser.add_argument('--log_graph', default=1, type=int,
                            help='Log compuational graph on tensorboard')
        parser.add_argument('--log_histogram', default=0, type=int,
                            help='Log histogram for weights and bias')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--learning_rate', default=2e-3, type=float)
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float,
                             help='weight decay (default: 1e-4)')
        parser.add_argument('--hidden_dim', type=int, default=3,
                            help='number of hidden dim for ConvLSTM layers')
        parser.add_argument('--ip_dim', type=int, default=3,
                            help='number of input dim for ConvLSTM layers')
        parser.add_argument('--num_workers', default=24, type=int)

        return parser
