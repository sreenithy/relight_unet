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
import cv2
from imageio import imsave
numbers = re.compile(r'(\d+)')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
RES = 256
from torchvision.utils import save_image
for img_path in glob.glob('s2/*.png'):

    img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
    l = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
    v = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[1])
    # print("in process",img_path,l,v)
    relit_mask = 'mask1/' + img_id + '_' + str(v) + '_' +str(l)+'.png'
    relit_mask = Image.open(relit_mask)
    relit_mask = relit_mask.resize((RES, RES))
    relit_mask = np.array(relit_mask)
    relit_mask = np.tile(relit_mask[:, :, np.newaxis], (1, 1, 3))

    ip = Image.open(img_path)
    ip = ip.resize((RES, RES))
    ip = np.array(ip)
    ip = cv2.bitwise_and(ip, relit_mask)
    imsave('mask1/'+img_id+'_'+str(v)+'_'+l+'.png',ip)