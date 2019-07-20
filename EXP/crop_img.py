import os
import os.path as osp

import cv2

# Read images
path = '/home/hejia/Projects/Robot-Interaction-Learning/data/youcook_0007_part0'

frame_idxes = []
for frame_name in os.listdir(path):
    frame_name = frame_name.replace('.jpg', '')
    frame_idx = int(frame_name)
    frame_idxes.append(frame_idx)
frame_idxes.sort()

frame_path_lst = []
for idx in frame_idxes:
    frame_path_lst.append(osp.join(path, '{}.jpg'.format(idx)))

# crop and write images
out_path = '/home/hejia/Projects/Robot-Interaction-Learning/data/youcook_0007_part0_cropped'
for i, frame_path in enumerate(frame_path_lst):
    img = cv2.imread(frame_path)
    img_cropped = img[16:, :, :]
    cv2.imwrite(osp.join(out_path, '{}.jpg'.format(i)),
                img_cropped)
