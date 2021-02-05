import scipy.io
import scipy.io

from utils.data_augmentation import *
import matplotlib.pyplot as plt
numbers = re.compile(r'(\d+)')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
RES = 256


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def modifyface(ip):
    ip = np.array(ip)
    #convert int to float
    ip = ip.astype(np.float32)
    #normalize to 0-1
    ip = ip/255
    # Gamma corrected image converted to linear space
    ip = np.power(ip, 2.2)
    # Range between 0 and 1
    ip = (ip - np.min(ip)) / (np.max(ip) - np.min(ip))
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip

def modifylight(ip):
    ip = np.array(ip)
    #convert int to float
    ip = ip.astype(np.float32)
    #normalize to 0-1
    ip = ip/255
    # Range between 0 and 1
    ip = (ip - np.min(ip)) / (np.max(ip) - np.min(ip))
    ip = ip.transpose((2, 0, 1))  # c,h,w
    ip = torch.from_numpy(ip).float()
    return ip

# def modifylight(ip):
#     ip = np.array(ip)
#     ip = ip.astype(np.float32) / 255.0
#     ip = ip.transpose((2, 0, 1))  # c,h,w
#     ip = np.clip(ip, 0, 1)
#     ip = torch.from_numpy(ip).float()
#     return ip


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


