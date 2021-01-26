
import os
import random
import glob
from collections import defaultdict
from numpy import asarray
import PIL
import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from skimage.transform import resize
import re
numbers = re.compile(r'(\d+)')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import cv2
def colour_jitter(ip, ip_light, op, op_light):
    bright_ip = torch.tensor(1.0).uniform_(0.7, 1).item()
    ip = TF.adjust_brightness(ip, bright_ip)
    ip_light = TF.adjust_brightness(ip_light, bright_ip)

    bright_op = torch.tensor(1.0).uniform_(0.7, 1).item()
    op = TF.adjust_brightness(op, bright_op)
    op_light = TF.adjust_brightness(op_light, bright_op)

    contrast_factor = torch.tensor(1.0).uniform_(0.7, 1).item()
    ip = TF.adjust_gamma(ip, contrast_factor)
    ip_light = TF.adjust_gamma(ip_light, contrast_factor)

    contrast_factor = torch.tensor(1.0).uniform_(0.7, 1).item()
    op = TF.adjust_gamma(op, contrast_factor)
    op_light = TF.adjust_gamma(op_light, contrast_factor)

    saturation_factor = torch.tensor(1.0).uniform_(0.7, 1).item()
    ip = TF.adjust_saturation(ip, saturation_factor)
    ip_light = TF.adjust_saturation(ip_light, saturation_factor)

    saturation_factor = torch.tensor(1.0).uniform_(0.7, 1).item()
    op = TF.adjust_saturation(op, saturation_factor)
    op_light = TF.adjust_saturation(op_light, saturation_factor)

    hue_factor = torch.tensor(1.0).uniform_(0, 0.05).item()
    ip = TF.adjust_hue(ip, hue_factor)
    ip_light = TF.adjust_hue(ip_light, hue_factor)

    hue_factor = torch.tensor(1.0).uniform_(0, 0.05).item()
    op = TF.adjust_hue(op, hue_factor)
    op_light = TF.adjust_hue(op_light, hue_factor)

    return ip, ip_light, op, op_light
