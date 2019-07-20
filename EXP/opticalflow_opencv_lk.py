import os
import os.path as osp

import numpy as np
import cv2

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

# params for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# create some random colors
color = np.random.randint(0, 255, (100, 3))

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
    # cv2.imwrite('{}.jpg'.format(i), gray_img)

old_gray = gray_img_lst[0]
old_frame = img_lst[0]
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

for i, (frame_gray, frame) in enumerate(zip(gray_img_lst[1:], img_lst[1:])):
    p1 = np.zeros_like(p0)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # p0r, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[j].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[j].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imwrite('lk_out/draw_{}.jpg'.format(i), img)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
