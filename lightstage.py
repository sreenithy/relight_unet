import scipy.io
import scipy.io
from utils.processdata import *
from utils.data_augmentation import *

numbers = re.compile(r'(\d+)')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
RES = 256


class LightStageFrames(Dataset):

    def __init__(self, path):
        # Get a list of { (identity+light) : filename }
        self.dataList = defaultdict(list)
        for f in sorted(glob.glob(str(path / "*.png")), key=numericalSort):
            self.dataList["_".join(f.split(".")[:-1])].append(f)
        self.dataKeys = list(self.dataList.keys())
        self.path = str(path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.44360004, 0.40193147, 0.31643618],
                                                                  std=[0.32258241, 0.29354254, 0.23349941])
                                             ])

    def __len__(self):
        return len(self.dataKeys)

    def _processlaval(self, img_path, op_path):
        img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
        l = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        v = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[1])

        ip = preprocess(img_path + '.png')
        name = str(img_path.split(".")[0])
        name = str(name.split("/")[1])
        op = preprocess(op_path + '.png')
        op_l = str(os.path.splitext(os.path.basename(op_path))[0].split("_")[-1])
        #Fix target light for all images
        if op_l != 'O9C4A044':
            l,op_l = op_l,l
            ip,op = op,ip
        ip_light = Image.open('envmap/' + img_id + '_' + str(v) + '_' + l + '.png')
        op_light = Image.open('envmap/'+ img_id + '_' + str(v) + '_' + op_l + '.png')
        ip, ip_light, op, op_light = colour_jitter(ip, ip_light, op, op_light)
        ip = modifyface(ip)
        ip_light = modifylight(ip_light)
        op = modifyface(op)
        op_light = modifylight(op_light)

        return ip, op, ip_light, op_light, l + '_' + str(v), op_l + '_' + str(v)

    def __getitem__(self, index):
        if self.path == 'train_X':
            divby = 2#28
        else:
            divby = 2#6
        img_path = self.dataKeys[index]
        op_cnd = 1
        factor = int(index / divby)
        # divby * factor + 1, divby * (factor + 1) - 2
        target_index = random.randint(divby * factor, divby * (factor + 1) - 1)
        while op_cnd:
            if target_index != index:
                op_cnd = 0
            else:
                target_index = random.randint(divby * factor,
                                              divby * (factor + 1) - 1)
        op_path = self.dataKeys[target_index]

        img_ip, img_op, ip_light, op_light, ip,op = self._processlaval(img_path, op_path)
        return img_ip, img_op, ip_light, op_light, ip,op
