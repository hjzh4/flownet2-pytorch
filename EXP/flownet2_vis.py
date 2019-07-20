import os
import os.path as osp

import cv2
import numpy as np

FLOWNET2_INF_OUT = '/home/hejia/Projects/flownet2-pytorch/out/inference/run.epoch-0-flow-field'


flo_idxes = []
for file_name in os.listdir(FLOWNET2_INF_OUT):
    file_name = file_name.replace('.flo', '')
    flo_idx = int(file_name)
    flo_idxes.append(flo_idx)
flo_idxes.sort()

flo_path_lst = []
for flo_idx in flo_idxes:
    flo_path_lst.append(osp.join(FLOWNET2_INF_OUT,
                                 '{}.flo'.format(1000000 + flo_idx)[1:]))
# read flow file
for i, flo_path in enumerate(flo_path_lst):
    with open(flo_path, 'rb') as f:
        x = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]

        data = np.fromfile(f, np.float32)
        data = np.reshape(data, newshape=(h, w, 2))
        data_x = data[:, :, 0]
        data_y = data[:, :, 1]

        mag, ang = cv2.cartToPolar(data_x, data_y)
        hsv = np.zeros(shape=(h, w, 3))
        hsv[:, :, 1] = 255
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = hsv.astype(np.float32)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('flownet_out/draw_{}.jpg'.format(i), bgr)



