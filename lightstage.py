# Returns  ip, op, ip_light, op_light, fullimgip
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
im_mul = [1.967532642055168, 2.1676019290963535, 2.747103618770585]
albedo_mul = [2.319305570225163, 2.353142937714471, 2.437376604762663]
RES = 256
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

import cv2
def colour_jitter(ip, ip_light, op, op_light):
    bright_ip = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    ip = TF.adjust_brightness(ip, bright_ip)
    ip_light = TF.adjust_brightness(ip_light, bright_ip)

    bright_op = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    op = TF.adjust_brightness(op, bright_op)
    op_light = TF.adjust_brightness(op_light, bright_op)

    contrast_factor = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    ip = TF.adjust_gamma(ip, contrast_factor)
    ip_light = TF.adjust_gamma(ip_light, contrast_factor)

    contrast_factor = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    op = TF.adjust_gamma(op, contrast_factor)
    op_light = TF.adjust_gamma(op_light, contrast_factor)

    saturation_factor = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    ip = TF.adjust_saturation(ip, saturation_factor)
    ip_light = TF.adjust_saturation(ip_light, saturation_factor)

    saturation_factor = torch.tensor(1.0).uniform_(0.7, 1.3).item()
    op = TF.adjust_saturation(op, saturation_factor)
    op_light = TF.adjust_saturation(op_light, saturation_factor)

    hue_factor = torch.tensor(1.0).uniform_(0, 0.05).item()
    ip = TF.adjust_hue(ip, hue_factor)
    ip_light = TF.adjust_hue(ip_light, hue_factor)

    hue_factor = torch.tensor(1.0).uniform_(0, 0.05).item()
    op = TF.adjust_hue(op, hue_factor)
    op_light = TF.adjust_hue(op_light, hue_factor)

    return ip, ip_light, op, op_light

def modify_mask(ip):
    ip = np.array(ip)
    ip = ip.astype(np.float32) / 255.0
    ip = np.tile(ip[:, :, np.newaxis], (1, 1, 3))
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip

def modifylight(ip):
    ip = np.array(ip)
    ip = ip.astype(np.float32) / 255.0
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip

def modify(ip, ratio):
    #ip = np.array(ip)
    ip = ip.astype(np.float32)/255
    ip = ip*ratio
    #ip = np.clip(ip, 0, 1)
    ip = cv2.normalize(ip, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip

def read(path):

    a = Image.open(path)
    a = np.array(a.resize((RES, RES)))
    return a

class LightStageFrames(Dataset):

    def __init__(self, path, transform=None):
        # Get a list of { (identity+light) : filename }
        self.dataList = defaultdict(list)
        # for f in sorted(glob.glob(str(path / "*.png")), key=numericalSort):
        for f in glob.glob(str(path / "*.png")):
            self.dataList["_".join(f.split("_")[:3])].append(f)

        self.dataKeys = list(self.dataList.keys())
        self.transform = transform
        self.path = str(path)
        myfile = open('mean_adjustment.txt', "rt")
        self.contents = myfile.read()
        myfile.close()

    def __len__(self):
        return len(self.dataKeys)


    def _processolat(self, img_path, RES):

        img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
        l = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        v = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[1])
        #print("in process",img_path,l,v)
        relit_mask = 'mask/' + img_id + '_' + str(v) + '.png'
        relit_mask = Image.open(relit_mask)
        relit_mask = relit_mask.resize((RES, RES))
        relit_mask = np.array(relit_mask)
        relit_mask = np.tile(relit_mask[:, :, np.newaxis], (1, 1, 3))

        ip = Image.open(img_path)
        ip = ip.resize((RES, RES))
        light_path = "lightolat/{}_{}.png".format(str(v), l)
        ip_light = Image.open(light_path)
        ip_light = ip_light.resize((32, 16))
        ip, ip_light, _, _ = colour_jitter(ip, ip_light, ip, ip_light)
        ip = np.array(ip)
        ip = cv2.bitwise_and(ip, relit_mask)
        # plt.imshow(ip);plt.show()

        ip = modify(ip, im_mul)
        ip_light = modifylight(ip_light)
        # mask_path = 'albedo_mask/' + img_id + '.png'
        mask_path = 'mask/' + img_id + '_' + str(v) +'.png'
        # full = 'Full/{}.png'.format(img_id)
        full = 'Fullview/' + img_id + '_' + str(v).zfill(2) + '.png'
        # print( img_path, full, light_path, mask_path, v, l)
        mask = Image.open(mask_path)
        mask = mask.resize((RES, RES))
        # relit_mask = np.tile(relit_mask[:, :, np.newaxis], (1, 1, 3))
        mask = modify_mask(mask)

        fullimgip = Image.open(full)
        fullimgip = fullimgip.resize((RES, RES))
        fullimgip = np.array(fullimgip)
        fullimgip = cv2.bitwise_and(fullimgip, relit_mask)
        fullimgip = modify(fullimgip, albedo_mul)
        return ip, [], ip_light, [], mask, fullimgip

    def _processlaval(self, img_path, RES):
        img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
        l = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        v = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[1])
        relit_mask = 'mask/' + img_id + '_' + str(v) + '.png'
        relit_mask = Image.open(relit_mask)
        relit_mask = relit_mask.resize((RES, RES))
        relit_mask = np.array(relit_mask)
        relit_mask = np.tile(relit_mask[:, :, np.newaxis], (1, 1, 3))

        #print("in process", img_path, l, v,'lightmat/' + l + '_' + str(v) + '.mat')
        ip = Image.open(img_path)
        ip = ip.resize((RES, RES))
        name = str(img_path.split(".")[0])
        name = str(name.split("/")[1])
        e = scipy.io.loadmat('lightmat/' + l + '_' + str(v) + '.mat')
        ip_light = self.lightprocess(e, name)

        ip, ip_light, _, _ = colour_jitter(ip, ip_light, ip, ip_light)
        ip = np.array(ip)
        ip = cv2.bitwise_and(ip, relit_mask)
        ip = modify(ip, im_mul)
        ip_light = modifylight(ip_light)

        # mask_path = 'albedo_mask/' + img_id + '.png'
        mask_path = 'mask/' + img_id + '_' + str(v) + '.png'
        full = 'Fullview/' + img_id +'_'+str(v).zfill(2)+'.png'

        # print( img_path, full, mask_path, v, l)
        mask = Image.open(mask_path)
        mask = mask.resize((RES, RES))
        mask = modify_mask(mask)

        fullimgip = Image.open(full)
        fullimgip = fullimgip.resize((RES, RES))
        fullimgip = np.array(fullimgip)
        fullimgip = cv2.bitwise_and(fullimgip, relit_mask)
        fullimgip = modify(fullimgip, albedo_mul)

        return ip, [], ip_light, [], mask, fullimgip

    def __getitem__(self, index):

        RES = 256
        img_path = self.dataList[self.dataKeys[index]]
        # l = str(os.path.splitext(os.path.basename(img_path[0]))[0])
        l = str(os.path.splitext(os.path.basename(img_path[0]))[0]).split("_")[-1]
        if l in ['10', '11', '13', '20', '21', '23', '24', '25', '26', '34', '35', '36', '37', '38', '39', '41',
                 '48', '49', '50', '53', '55', '56', '57', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                 '71', '72', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '89', '90', '91', '94',
                 '95', '96', '97', '98', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '114',
                 '115', '116', '117', '118', '119', '120', '121', '122', '123', '125', '126', '127', '128', '129',
                 '130', '131', '133', '134', '135', '136', '137', '138', '141', '142', '143', '144', '146','147', '149']:
            img, _, light, _, mask, fullimg = self._processolat(img_path[0], RES)
        else:
            img, _, light, _, mask, fullimg = self._processlaval(img_path[0], RES)

        return img, [], light, [], fullimg,mask

    def lightprocess(self, e, name):
        #print("e",name)
        regex = r"^\b(?=\w)" + re.escape(name) + ".\d+\S+"
        matches = re.finditer(regex, self.contents, re.MULTILINE)
        for matchNum, match in enumerate(matches, start=1):
            line = match.group()
            mean_adjustment = float(line.split("\t")[-1])
            img = e['e'] * mean_adjustment
            img = np.clip(img, 0, 1) ** 0.4545
            img = np.clip(255. * img, 0, 255)
            ip_light = resize(img, (16, 32))
            ip_light = Image.fromarray(ip_light.astype('uint8'))
        return ip_light
