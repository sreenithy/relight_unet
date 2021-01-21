import torch
import torch.nn.functional as F
@torch.no_grad()
def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value*valid_mask
    if reduction == 'mean':
        return torch.mean(value)
    return value

@torch.no_grad()
def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def tv_loss(img, weight=1):
    # https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def loss_tv(self, mask, y_comp):
    '''
    #https://github.com/ryanwongsa/Image-Inpainting/blob/master/src/loss/loss_compute.py
    Args:
        self:
        mask:
        y_comp:

    Returns:

    '''
    kernel = torch.ones((3, 3, mask.shape[1], mask.shape[1]), requires_grad=False).to(self.device)
    dilated_mask = F.conv2d(1 - mask, kernel, padding=1) > 0
    dilated_mask = dilated_mask.clone().detach().float()

    P = dilated_mask * y_comp

    a = self.l1(P[:, :, :, 1:], P[:, :, :, :-1])
    b = self.l1(P[:, :, 1:, :], P[:, :, :-1, :])
    return a + b

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

