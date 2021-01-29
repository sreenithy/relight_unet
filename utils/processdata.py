import scipy.io
import scipy.io

from utils.data_augmentation import *

numbers = re.compile(r'(\d+)')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# im_mul = [1.967532642055168, 2.1676019290963535, 2.747103618770585]
# albedo_mul = [2.319305570225163, 2.353142937714471, 2.437376604762663]
RES = 256


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


#
# def modify_mask(ip):
#     ip = np.array(ip)
#     ip = ip.astype(np.float32) / 255.0
#     ip = np.tile(ip[:, :, np.newaxis], (1, 1, 3))
#     ip = ip.transpose((2, 0, 1))  # c,h,w
#     ip = torch.from_numpy(ip).float()
#     return ip

def modifyface(ip):
    ip = np.array(ip)
    ip = ip.astype(np.float32) / 255.0
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = np.power(ip,2.2) #Face converted to linear
    ip = torch.from_numpy(ip).float()
    return ip

def modifylight(ip):
    ip = np.array(ip)
    ip = ip.astype(np.float32) / 255.0
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip


def preprocess(path):
    a = Image.open(path)
    a = a.resize((RES, RES))
    return a

def modifytestface(ip):
    ip = np.array(ip)
    ip = ip[:,:,:3]
    ip = ip.astype(np.float32)/255
    ip = ip.transpose((2, 0, 1))
    ip = np.power(ip, 2.2)
    ip = ip[None, ...]
    ip = torch.from_numpy(ip).float()
    return ip


def modifytestlight(path):
    ip = Image.open(path)
    ip = np.array(ip)
    ip = ip[:,:,:3]
    ip = ip.astype(np.float32)/255
    ip = ip.transpose((2, 0, 1))
    ip = ip[None, ...]
    ip = torch.from_numpy(ip).float()
    return ip


