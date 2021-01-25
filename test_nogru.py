from lightstage import *
from PIL import Image
import cv2
import numpy as np
import re
import glob
from lightstage import LightStageFrames
import sys
sys.path.append('core')
from model import HourglassNet
from skimage.io import imread, imsave
from skimage import exposure
from pathlib import Path
numbers = re.compile(r'(\d+)')
device = torch.device("cpu")

if not os.path.exists('1'):
    os.makedirs('1')
if not os.path.exists('2'):
    os.makedirs('2')
def obtainimg(dataset_test,index):
    RES = 256
    img_paths = dataset_test.dataKeys[index]
    images = []
    cnt = 0
    img_pathslist = []
    for img_path in dataset_test.dataList[img_paths]:
        img_pathslist.append(img_path)
        img = process(img_path, RES)
        images.append(img)
        cnt += 1
    return images, img_pathslist

def process( img_path, RES):
    ip = Image.open(img_path)
    ip = ip.resize((RES, RES))
    ip = modify(ip)
    return ip

def normlight(image,  filename):
    image = image[0].cpu().data.numpy()
    image = image.transpose((1, 2, 0))
    image = image*255
    imsave(filename, image)

def normaliseimg(image, filename):
    image = image[0].cpu().data.numpy()
    a = image.transpose((1, 2, 0))
    imsave(filename, np.clip(255. * a, 0, 255).astype('uint8'))


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def modify(ip):
    ip = np.array(ip)
    ip = ip[:,:,:3]
    ip = ip.astype(np.float32)/255
    ip = ip.transpose((2, 0, 1))
    ip = ip[None, ...]
    ip = torch.from_numpy(ip).float()
    return ip

dataset_test = LightStageFrames(Path("s2/"))
length = len(dataset_test.dataList)
print(length)
RES=256
chk_pt = 'tb_logs/relightnet/version_101/checkpoints/epoch=413.ckpt'#tb_logs/relightnet/version_30/checkpoints/epoch=361.ckpt'
net = HourglassNet.load_from_checkpoint(chk_pt).to(device)
net.eval()

for idx in range(len(dataset_test)):
    inputs, img_paths = obtainimg(dataset_test, idx)
    for i, (input_,img_path) in enumerate(zip(inputs,img_paths)):
        print(img_path)
        IP = input_.to(device)
        albedo_estim, light_estim = net(IP)
        name = (os.path.splitext(os.path.basename(img_path))[0].split(".")[0])
        normaliseimg(light_estim,  '1/' + name +'.png')
        normaliseimg(albedo_estim,  's2/' + name + '_relit.png')
