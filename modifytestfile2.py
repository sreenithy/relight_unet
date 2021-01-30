import sys

from lightstage import *
from skimage.io import imsave, imread
sys.path.append('core')
numbers = re.compile(r'(\d+)')
device = torch.device("cpu")


def preprocess(path):
    ip= Image.open(path)
    ip = np.array(ip)
    ip = ip.astype(np.float32) #/ 255.0
    ip  = np.power(ip,2.2)
    plt.imshow(ip);plt.show()
    ip =np.power(ip,0.4545)
    plt.imshow(ip); plt.show()


x = preprocess('val_s/s080-041119-02_2_O9C4A034.png')
