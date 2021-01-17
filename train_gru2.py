import sys
sys.path.append('core')
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
parser = argparse.ArgumentParser()
from model import HourglassNet
from argparse import ArgumentParser
from torchsummary import summary
import torch
def main(hparams):

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    net = HourglassNet(hparams=hparams)
    # checkpoint = torch.load("tb_logs/relightnet/version_29/checkpoints/epoch=1788.ckpt")
    # net.load_state_dict(checkpoint["state_dict"])
    logger = TensorBoardLogger("tb_logs", name="relightnet")
    #logger2 = CSVLogger('Csv')

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(max_epochs=1000000,
                         gpus=hparams.gpus,
                         prepare_data_per_node=False,
                         profiler=False,
                         distributed_backend=hparams.distributed_backend,
                         logger=logger,gradient_clip_val=0.5, replace_sampler_ddp=False,resume_from_checkpoint='tb_logs/relightnet/version_34/checkpoints/epoch=2399.ckpt')
    # trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20)
    # trainer.global_step = checkpoint["global_step"]
    # trainer.current_epoch = checkpoint["epoch"]

    # ------------------------
    # 3 FIND LR
    # ------------------------
    # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(net)

    # # Results can be found in
    # print(lr_finder.results)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print("new_lr", new_lr)

    # # update hparams of the model
    # net.hparams.learning_rate = new_lr
    # print(new_lr)


    # ------------------------
    # 4 START TRAINING
    # ------------------------
    # find the batch size

    trainer.fit(net)
    #net.to('cuda')
    #summary(net, [(3, 256, 256), (3,16,32)])

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    log_dir = 'tmp'
    parser = HourglassNet.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
    main(hyperparams)
