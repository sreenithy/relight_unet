from lightstage import *
from PIL import Image
import cv2

import numpy as np
import re
import glob

import sys
sys.path.append('core')
from model import HourglassNet
numbers = re.compile(r'(\d+)')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import faulthandler
faulthandler.enable()

def normaliseimg(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    norm_image.astype(np.uint8)
    return norm_image


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def modify(ip):
    ip = np.array(ip)
    ip = ip.astype(np.float32) / 255.0
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = ip[None, ...]
    ip = torch.from_numpy(ip)
    return ip


chk_pt = 'epoch=7242.ckpt'
net = HourglassNet.load_from_checkpoint(chk_pt).to(device)
net.eval()
for f in sorted(glob.glob('test_light/*'), key=numericalSort):
    ipall=None
    lightall=None

    for test_img_path in sorted(glob.glob('test/*.png'), key=numericalSort):
        name = test_img_path[5:-4]
        print(name)
        print("Light name", f[13:])
        ip = modify(Image.open(test_img_path))
        op_light = modify(Image.open(f))
        if ipall is None:
            ipall=ip
        else:
            ipall = torch.cat([ipall, ip_net], 0)
        if lightall is None:
            lightall=op_light
        else:
            lightall = torch.cat([lightall, op_light], 0)

        print(ip.shape, ipall.shape, op_light.shape,lightall.shape,type(lightall))
    ipall, lightall = ipall.to(device), lightall.to(device)
    outputImg, est_light,est_fullface,hiddens = net(ipall, lightall,0,None)

    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)

    outputImg1 = normaliseimg(outputImg[:,:,0:3])
    outputImg1 = outputImg1[..., [2, 1, 0]].copy()
    filename='relit/'+name+'_relit_lighting' + f[12:]
    print(filename)
    cv2.imwrite(filename, outputImg1)
    est_light=est_light[0].cpu().data.numpy()
    light = est_light.transpose((1, 2, 0))
    light = np.squeeze(light)
    light = normaliseimg(light)
    light = light[..., [2, 1, 0]].copy()
    cv2.imwrite('relit/'+name+f[13:15]+'_extracted_light.png', light)

    est_fullface = est_fullface[0].cpu().data.numpy()
    est_fullface = est_fullface.transpose((1, 2, 0))
    est_fullface = np.squeeze(est_fullface)
    est_fullface = normaliseimg(est_fullface)
    est_fullface = est_fullface[..., [2, 1, 0]].copy()
    cv2.imwrite('relit/' + name+f[13:15] + '_extracted_full_texture.png', est_fullface)
