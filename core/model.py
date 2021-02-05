import sys
from argparse import ArgumentParser
from pathlib import Path
from torch.autograd import Variable
sys.path.append('functions')
sys.path.append('utils')
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image
from utils.metrics import *
import os
#import pytorch_msssim
#m = pytorch_msssim.MSSSIM()
from functions.func import *

if not os.path.exists('results_face'):
    os.makedirs('results_face')

import pytorch_lightning as pl
from lightstage import *
from update import *
from lighting import lightingNet
from up_new import *

from envmap import EnvironmentMap
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


class RelightNetwork(pl.LightningModule):

    def __init__(self, hparams):
        super(RelightNetwork, self).__init__()
        # self.hparams = hparams
        self.hparams = hparams
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.log_images = hparams.log_images
        self.weight_decay = hparams.weight_decay
        self.num_workers = hparams.num_workers
        self.bilinear = hparams.bilinear

        self.psnrtable = pd.DataFrame()
        self.layer0 = preconv()
        self.layer1 = Down(32, 64, Pad=1, Stride=2)
        self.layer2 = Down(64, 128, Pad=1, Stride=2)
        self.layer3 = Down(128, 256, Pad=1, Stride=2)
        self.layer4 = TripleConv(256, 512)
        self.lightingNet = lightingNet(512)
        self.layer5 = TripleUp(1024, 512)
        self.layer6 = Up(512, 256)
        self.layer7 = Up(256, 128)
        self.layer8 = Up(128, 64)

        # self.layer9 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(32),
            nn.PReLU())
        self.layer10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.sa = EnvironmentMap(16, 'LatLong').solidAngles()
        self.save_hyperparameters()

    def forward(self, x, target_light):

        x0 = self.layer0(x)
        x10, x11 = self.layer1(x0)
        x20, x21 = self.layer2(x11)
        x30, x31 = self.layer3(x21)
        x40, x41, x42 = self.layer4(x31)
        x5, light_estim = self.lightingNet(x42, target_light)

        x6 = self.layer5(x5, x40, x41, x42)
        x6 = self.layer6(x6, x30, x31)
        x6 = self.layer7(x6, x20, x21)
        x6 = self.layer8(x6, x10, x11)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x0, x6], dim=1)
        x6 = self.layer9(x6)
        face_estim = self.layer10(x6)
        face_estim = torch.sigmoid(face_estim)
        return face_estim, light_estim

    def training_step(self, batch, batch_nb):
        # self.trainer.optimizers[0].param_groups[-1]['lr'] = 1e-3
        input_, output_face, light_input, light_output, ip,op = batch
        face_estim, light_estim = self.forward(input_, light_output)
        weights = Variable(torch.from_numpy(self.sa), requires_grad=False).float().to(self.lightingNet.device)
        sz = output_face.size(2) ** 2
        l1_face =10/sz*torch.sum(torch.abs(face_estim - output_face)) / face_estim.shape[0]
        wl2 = 0.08*torch.sum(weights * torch.abs(torch.log(1+light_estim) - torch.log(1+light_input)))**2 / face_estim.shape[0]

        # l_msssim = (1 - m(output_face, face_estim)) * 1000

        loss = l1_face + wl2
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).to(self.lightingNet.device)

        if self.log_images:
            if self.global_step % 10 == 0:
                light_estim = F.interpolate(light_estim, (128, 256), mode="bilinear")
                light_input = F.interpolate(light_input, (128, 256), mode="bilinear")
                light_output = F.interpolate(light_output, (128, 256), mode="bilinear")
                light_diff = 3 * torch.abs(light_estim[0:1, ...] - light_input[0:1, ...])
                light_diff = F.interpolate(light_diff, (128, 256), mode="bilinear")

                # light_diff = F.pad(light_diff, [0 // 2, 0 - 0 // 2, 128 // 2, 128 - 128 // 2])
                light_output = F.pad(light_output, [0 // 2, 0 - 0 // 2, 128 // 2, 128 - 128 // 2])
                light_ip256 = F.pad(light_input, [0 // 2, 0 - 0 // 2, 128 // 2, 128 - 128 // 2])
                light_estim256 = F.pad(light_estim, [0 // 2, 0 - 0 // 2, 128 // 2, 128 - 128 // 2])
                light_diff256 = F.pad(light_diff, [0 // 2, 0 - 0 // 2, 128 // 2, 128 - 128 // 2])

                face_diff = 3 * torch.abs(face_estim[0:1, ...] - output_face[0:1, ...])

                img_stack = torch.clamp(
                    torch.cat((input_[0:1, ...], output_face[0:1, ...], face_diff, face_estim[0:1, ...],
                               light_ip256[0:1, ...], light_output[0:1, ...], light_diff256, light_estim256[0:1, ...]
                               )), min=0, max=1)
                albedo_grid = make_grid_with_lightlabels(img_stack.detach().cpu(),
                                                         ["Input", "Target", "Diff (3x)", "Target estim",
                                                         "Input Light ", "Target Light", "Diff (3x)", " Input Estim"],
                                                         nrow=4)
                save_image(albedo_grid,
                           'results_face/epoch_{}_step_{}_face_images.png'.format(self.current_epoch,
                                                                                  self.global_step))


                plt.close()
        if self.hparams.log_graph == 1:
            # Logging the computational graph on tensorboard
            if self.global_step == 1:
                example_input_array = list()
                example_input_array.append(torch.rand((1, 3, 256, 256)))
                self.logger.experiment.add_graph(RelightNetwork(self.hparams), example_input_array)
                print("Logged computational Graph")

        if self.hparams.log_histogram == 1:
            self.custom_histogram_adder()

        self.log('l1_face', l1_face, prog_bar=True)
        self.log('wl2', wl2, prog_bar=True)
        self.log('learning_rate', lr_saved, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        print("Validation")
        input_, output_face, light_input, light_output, ip,op = batch

        face_estim, light_estim = self.forward(input_, light_output)

        face_estim = torch.clamp(face_estim, min=0, max=1)
        light_estim = torch.clamp(light_estim, min=0, max=1)
        weights = Variable(torch.from_numpy(self.sa), requires_grad=False).float().to(self.lightingNet.device)
        # Calculate loss
        sz = output_face.size(2) ** 2
        l1_face = 10 / sz * torch.sum(torch.abs(face_estim - output_face)) / face_estim.shape[0]
        wl2 = 0.08*torch.sum(weights * torch.abs(torch.log(1 + light_estim) - torch.log(1 + light_input))) ** 2 / \
              face_estim.shape[0]

        loss = l1_face + wl2

        psnr_ = psnr(image_pred=face_estim, image_gt=output_face) / face_estim.shape[0]

        self.log('val_loss', loss)
        self.log('val_loss', loss, prog_bar=True)

        return {
            "val_loss": loss,
            "val_psnr": psnr_
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_valloss', avg_loss, prog_bar=True)
        return {'avg_valloss': avg_loss}

    def psnr_table(self, outputs, tag):
        psnr = []
        for i in range(len(outputs)):
            psnr.append(outputs[i][tag])
        avg_psnr = np.nanmean(np.array(list(zip_longest(*psnr)), dtype=float), axis=1)
        avg_psnr = avg_psnr.reshape(1, len(avg_psnr))
        avg_psnr = pd.DataFrame(avg_psnr)
        filename = self.logger.root_dir + '/version_' + str(self.logger.version - 1) + '/' + tag + '.csv'
        avg_psnr.to_csv(filename, mode='a', sep=' ', header=False, )

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     betas=(0.5, 0.999))

        scheduler = MultiStepLR(optimizer, milestones=[1790, 4000], gamma=0.5)
        # scheduler =  MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.5)
        return [optimizer], [scheduler]

    def __dataloader(self):
        dataset_train = LightStageFrames(Path("train_X/"))
        dataset_val = LightStageFrames(Path("val_X/"))
        train_loader = FastDataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                      pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        return {'train': train_loader, 'val': val_loader}

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
        parser.add_argument('--log_graph', default=0, type=int,
                            help='Log computational graph on tensorboard')
        parser.add_argument('--log_histogram', default=0, type=int,
                            help='Log histogram for weights and bias')
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--learning_rate', default=5e-5 , type=float)
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--bilinear', default=0, type=int,
                            help='upsampling')
        parser.add_argument('--num_workers', default=24, type=int)

        return parser
