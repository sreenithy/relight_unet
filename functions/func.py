import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch
import pandas as pd
import numpy as np
from itertools import zip_longest

im_mul = [1.9425111494312444, 2.173566784839146, 3.159374388266134]
albedo_mul = [2.2330479393857896, 2.3364531836931643, 2.7321869878173968]

def makegrid(output, numrows):
    outer = (torch.Tensor.cpu(output).detach())
    plt.figure(figsize=(20, 5))
    b = np.array([]).reshape(0, outer.shape[2])
    c = np.array([]).reshape(numrows * outer.shape[2], 0)
    i = 0
    j = 0
    while (i < outer.shape[1]):
        img = outer[0][i]
        b = np.concatenate((img, b), axis=0)
        j += 1
        if (j == numrows):
            c = np.concatenate((c, b), axis=1)
            b = np.array([]).reshape(0, outer.shape[2])
            j = 0

        i += 1
    return c


import torch
import math
from torchvision.transforms import transforms
import cv2
irange = range


# Taken from https://discuss.pytorch.org/t/add-label-captions-to-make-grid/42863/4
def make_grid_with_lightlabels(tensor, labels, nrow=10, limit=20, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if not isinstance(labels, list):
        raise ValueError
   # else:
    #    #import pdb; pdb.set_trace()
    #    labels = np.asarray(labels)#.T[0]
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]

    import cv2
    font = 1
    fontScale = 2
    color = (255, 255, 255)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)

    k=0
    for y in irange(ymaps):
        for x in irange(xmaps):

            if k >= nmaps:
                break
            working_tensor = tensor[k]

            if labels is not None:
                org = (0, int(tensor[k].shape[1] * 0.9))

                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.numpy(), (1, 2, 0)) * 255  ).astype('uint8'))

                image = cv2.putText(working_image, f'{str(labels[k])}', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(image.get())

            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid

# Taken from https://discuss.pytorch.org/t/add-label-captions-to-make-grid/42863/4
def make_grid_with_labels(tensor, labels, nrow=10, limit=20, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if not isinstance(labels, list):
        raise ValueError
   # else:
    #    #import pdb; pdb.set_trace()
    #    labels = np.asarray(labels)#.T[0]
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]

    import cv2
    font = 1
    fontScale = 2
    color = (255, 255, 255)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)

    k=0
    for y in irange(ymaps):
        for x in irange(xmaps):

            if k >= nmaps:
                break
            working_tensor = tensor[k]

            if labels is not None:
                org = (0, int(tensor[k].shape[1] * 0.9))
                '''
                
                if k/2 ==1:
                    mean = [0.09206585, 0.07940055, 0.05346605]
                    std = [0.1682443, 0.14520547, 0.09966198]
                    ratio = [5.430895386291442, 6.2971855988403105, 9.35172880734597]

                else:
                    mean = [0.23182251, 0.21877623, 0.18769414]
                    std = [0.28310018, 0.26406067, 0.22603413]
                    ratio = [2.156822475953694, 2.2854402418398014, 2.663908420369437]
                '''

                ratio = [2.319305570225163, 2.353142937714471, 2.437376604762663]
                a = np.transpose(working_tensor.numpy(), (1, 2, 0))*255
                a /= ratio
                a = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                a = a.astype(np.uint8)

                working_image = cv2.UMat(a)

                image = cv2.putText(working_image, f'{str(labels[k])}', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(image.get())

            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid
