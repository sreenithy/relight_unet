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
        myfile = open('mean_adjustment.txt', "rt")
        self.contents = myfile.read()
        myfile.close()
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

        # e = scipy.io.loadmat('lightmat/' + l + '_' + str(v) + '.mat')
        # ip_light = self.lightprocess(e, name)
        # op_l = str(os.path.splitext(os.path.basename(op_path))[0].split("_")[-1])
        # e_op = scipy.io.loadmat('lightmat/' + op_l + '_' + str(v) + '.mat')
        # op_light = self.lightprocess(e_op, name)

        op_l = str(os.path.splitext(os.path.basename(op_path))[0].split("_")[-1])
        # #
        # ip_light = Image.open('gtlg/'+ l + '_' + str(v) +'.png')
        # op_light = Image.open('gtlg/'+ op_l + '_' + str(v) + '.png')

        ip_light = Image.open('lavalmapsall/' + img_id+'_' + str(v) +'_'+l +  '.png')
        op_light = Image.open('lavalmapsall/'+ img_id+'_' + str(v) +'_'+ op_l  + '.png')


        ip, ip_light, op, op_light = colour_jitter(ip, ip_light, op, op_light)
        ip = np.array(ip)
        ip = modifyface(ip)
        ip_light = modifylight(ip_light)
        op = modifyface(op)
        op_light = modifylight(op_light)
        # print(img_path,op_path,'lightmat/' + l + '_' + str(v) + '.mat','lightmat/' + op_l + '_' + str(v) + '.mat')

        return ip, op, ip_light, op_light, l + '_' + str(v), op_l + '_' + str(v)

    def __getitem__(self, index):
        if self.path == 'train':
            divby = 28
        else:
            divby = 8
        img_path = self.dataKeys[index]
        op_cnd = 1
        factor = int(index / divby)
        # print(index, img_path, factor,self.dataKeys[divby * factor], self.dataKeys[divby * (factor + 1) - 1],divby*factor+1, divby*(factor+1)-2)
        # divby * factor + 1, divby * (factor + 1) - 2
        target_index = random.randint(divby * factor, divby * (factor + 1) - 1)
        while op_cnd:
            if target_index != index:
                op_cnd = 0
            else:
                target_index = random.randint(divby * factor,
                                              divby * (factor + 1) - 1)
        # print(target_index)
        op_path = self.dataKeys[target_index]

        img_ip, img_op, ip_light, op_light, ip,op = self._processlaval(img_path, op_path)
        # img = self.transform(img)
        return img_ip, img_op, ip_light, op_light, ip,op

    def lightprocess(self, e, name):
        global ip_light
        regex = r"^\b(?=\w)" + re.escape(name) + ".\d+\S+"
        matches = re.finditer(regex, self.contents, re.MULTILINE)

        for matchNum, match in enumerate(matches, start=1):
            line = match.group()
            mean_adjustment = float(line.split("\t")[-1])
            img = e['e'] * mean_adjustment
            img = np.clip(img, 0, 1)  # ** 0.4545
            #Removed gamma correction
            img = np.clip(255. * img, 0, 255)
            ip_light = resize(img, (16, 32))
            ip_light = Image.fromarray(ip_light.astype('uint8'))
        if ip_light is None:
            print(name)
        return ip_light
