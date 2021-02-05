import glob
import os
import re
import numpy as np
import scipy.io
from envmap import EnvironmentMap
from joblib import Parallel, delayed
from scipy import linalg
from skimage.io import imsave
from skimage.transform import resize

numbers = re.compile(r'(\d+)')
myfile = open('mean_adjustment.txt', "rt")
contents = myfile.read()
myfile.close()
view2rot = {2: 0, 3: 1, 6: 2, 7: 3, 10: 4, 11: 5}
sao = EnvironmentMap(4096, 'LatLong').solidAngles()
sat = EnvironmentMap(8, 'LatLong').solidAngles()
from scipy.io import savemat


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def listid():
    '''

    Returns:List of image id

    '''
    ID = []
    for img_path in sorted(glob.glob('relight/*'), key=numericalSort):
        img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
        if img_id not in ID:
            ID.append(img_id)
    return ID


def downscaleEnvmap(nenvmap, sao, sat, times):
    """Energy-preserving environment map downscaling by factors of 2.
    Usage:
    sao = EnvironmentMap(512, 'LatLong').solidAngles() # Source envmap solid angles, could be replaced by `sao = envmap.soldAngles()`
    sat = EnvironmentMap(128, 'LatLong').solidAngles() # Target envmap solid angles
    downscaleEnvmap(envmap, sao, sat, 3)
    Note : `times` is the number of downscales, so the total downscaling factor is 2**times"""
    nenvmap.data *= sao[:, :, np.newaxis]
    nenvmap.data = np.pad(nenvmap.data, [(0, 1), (0, 1), (0, 0)], 'constant')
    sx = np.cumsum(nenvmap.data, axis=1)
    tmp = sx[:, 2 ** times::2 ** times, ...] - sx[:, :-2 ** times:2 ** times, ...]
    sy = np.cumsum(tmp, axis=0)
    nenvmap.data = sy[2 ** times::2 ** times, :, ...] - sy[:-2 ** times:2 ** times, :, ...]
    if nenvmap.data.shape[1] > 2 * nenvmap.data.shape[0]:
        nenvmap.data[:, -2, :] += nenvmap[:, -1, :]
        nenvmap.data = nenvmap.data[:, :-1, :]
    nenvmap.data /= sat[:, :, np.newaxis]
    nenvmap.data = np.ascontiguousarray(nenvmap.data)
    return nenvmap


def obtainrotmat(img_id):
    rot = []
    for view in [2, 3, 6, 7, 10, 11]:
        f = open("/media/sreenithy/sree4tb/MERLDATASET-FULL/Lightstage_unzip_fixed/" + img_id + "/cam-calib/cam_" + str(
            view).zfill(2) + ".stv", "r")
        f1 = f.readline();
        f2 = f.readline();
        f3 = f2.replace("[", "");
        f3 = f3.replace("]", "");
        f3 = f3.replace("projection_matrix", "")
        f3 = f3.replace(",", "");
        f3 = f3.replace(";", "");
        f3 = f3.replace("   ", " ");
        f3 = f3.replace("  ", " ");
        f3 = f3.replace("\n", " ")
        up = 0
        for i in range(1,10):
            # print(f3.split(" ")[i])
            try:
                float(f3.split(" ")[i])
                up = up+1
            except ValueError:
                up = 100
        if up < 100:
            cam = np.empty((3, 3))
            cnt = 1
            for i in range(3):
                for j in range(3):
                    cam[i, j] = float(f3.split(" ")[cnt])
                    cnt = cnt + 1
            _, q = linalg.rq(cam)
            rot.append(q)
    return rot


def lightprocess(e, name):
    regex = r"^\b(?=\w)" + re.escape(name) + ".\d+\S+"
    matches = re.finditer(regex, contents, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        line = match.group()
        mean_adjustment = float(line.split("\t")[-1])
        img = e['e'] * mean_adjustment
        img = np.clip(img, 0, 1)
        img = np.clip(255. * img, 0, 255)
        ip_light = resize(img, (16, 32))
        ip_light_linear = ip_light.astype('uint8')
        imsave('alllaval_png/' + name + '.png', ip_light_linear)


def obtainlavalmat(rot, img_id, light, view):
    e = EnvironmentMap('/media/sreenithy/Mybook/adobe/Laval-work/envmap_process/allhdr/' + light + '.exr', 'latlong')
    e = e.resize(4096)
    e = downscaleEnvmap(e, sao, sat, 9)
    e = e.rotate('DCM', input_=rot)
    savemat('alllaval/' + img_id + '_' + str(view) + '_' + light + '.mat', {'e': e.data}, oned_as='row')
    name = img_id + '_' + str(view) + '_' + light
    e = scipy.io.loadmat('alllaval/' + img_id + '_' + str(view) + '_' + light + '.mat')
    lightprocess(e, name)


def processmap(idx):
    rotmat = obtainrotmat(idx)
    for img_path in sorted(glob.glob('relight/' + idx + '*'), key=numericalSort):
        img_id = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[0])
        l = str(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        v = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[1])
        obtainlavalmat(rotmat[view2rot[v]], img_id, l, v)


ID = listid()
# rotmat = obtainrotmat(ID[0])
r = Parallel(n_jobs=-1, verbose=1)([delayed(processmap)(ID[x]) for x in range(33,len(ID))])
