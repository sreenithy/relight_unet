
import os
import random
import glob
import scipy.io
import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
import re
from skimage.io import imsave
numbers = re.compile(r'(\d+)')

RES = 256
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

import cv2

myfile = open('mean_adjustment.txt', "rt")
contents = myfile.read()
myfile.close()
def lightprocess(e, name):

    img = e['e'] #*2.050137e+01
    img = np.clip(img, 0, 1) ** 0.4545
    img = np.clip(255. * img, 0, 255)
    ip_light = resize(img, (16, 32))
    ip_light = ip_light.astype('uint8')
    imsave('gtlg/'+ name +'.png', ip_light)


for img_path in sorted(glob.glob('lightmat/*.mat'), key=numericalSort):
    e = scipy.io.loadmat(img_path)
    name = str(img_path.split(".")[0])
    name = str(name.split("/")[1])
    lightprocess(e, name)