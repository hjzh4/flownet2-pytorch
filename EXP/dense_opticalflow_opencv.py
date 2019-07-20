import os
import os.path as osp

import cv2
import numpy as np

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

img_lst = []
gray_img_lst = []
for i, frame_path in enumerate(frame_path_lst):
    img = cv2.imread(frame_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_lst.append(gray_img)
    img_lst.append(img)

old_gray = gray_img_lst[0]
old_frame = img_lst[0]

# create a mask image for drawing purposes
hsv = np.zeros_like(old_frame)
hsv[:, :, 1] = 255

for i, (frame_gray, frame) in enumerate(zip(gray_img_lst[1:], img_lst[1:])):
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite('dense_out/draw_{}.jpg'.format(i), bgr)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
