from lightstage import *
from PIL import Image
import cv2
import numpy as np
import re
import glob
from lightstage import LightStageFrames
import sys
sys.path.append('core')
from model import RelightNetwork
from skimage.io import imread, imsave
from skimage import exposure
from pathlib import Path
numbers = re.compile(r'(\d+)')
device = torch.device("cpu")
from utils.processdata import *

if not os.path.exists('testface'):
    os.makedirs('testface')
if not os.path.exists('testlight'):
    os.makedirs('testlight')

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
    ip = modifytestface(ip)
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



dataset_test = LightStageFrames(Path("biden/"))
length = len(dataset_test.dataList)
print(length)
RES=256
chk_pt = 'tb_logs/relightnet/version_64/checkpoints/epoch=729.ckpt'#tb_logs/relightnet/version_30/checkpoints/epoch=361.ckpt'
net = RelightNetwork.load_from_checkpoint(chk_pt).to(device)
net.eval()
target_light = modifytestlight('lavalmapsall/s878-060714-02_6_O9C4A044.png')
for idx in range(len(dataset_test)):
    inputs, img_paths = obtainimg(dataset_test, idx)
    for i, (input_,img_path) in enumerate(zip(inputs,img_paths)):
        print(img_path)
        IP = input_.to(device)
        target_light = target_light.to(device)
        face_estim, light_estim = net(IP,target_light)
        name = (os.path.splitext(os.path.basename(img_path))[0].split(".")[0])

        imsave('testface/' + name +'.png',np.asarray(np.transpose(light_estim[0].cpu().data.numpy(), (1, 2, 0))* 255).astype('uint8'))
        imsave( 'testlight/' + name + '_relit.png',np.asarray(np.transpose(face_estim[0].cpu().data.numpy(), (1, 2, 0)) * 255).astype('uint8'))
