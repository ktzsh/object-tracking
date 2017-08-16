import utils.det_utils as util
import scripts.model as model_h

import cv2
import numpy as np
import math
import random
import time
import sys
import os

thresh = 0.8
hmap_size = 64
flag = False
y_predict = None

data_dirs = ['Human8']
data_dirs = ['Human3','Jogging-2', 'Skater', 'Jump', 'Girl2', 'Dancer']

def evaluate_dir(y_predict, data_dir, frame_bboxs, flag=False):

    path_prefix = './data/TB-50/'
    i, score, frames_total = 6, 0.0, 0
    x1_pred, y1_pred, x2_pred, y2_pred = None, None, None, None
    x1_true, y1_true, x2_true, y2_true = None, None, None, None

    for sample in y_predict:

        i = i + 1
        heat_map_vec = sample[5]
        heat_map = heat_map_vec.reshape((hmap_size, hmap_size))

        frame_path = path_prefix + data_dir.split('-')[0] + '/img/' + str(i).zfill(4) + '.jpg'
        frame = cv2.imread(frame_path)
        height, width, channels = frame.shape

        x1, y1, x2, y2 = util.generate_rectangle_from_heatmap(heat_map, thresh=thresh)

        ratio_y = float(height)/hmap_size
        ratio_x = float(width)/hmap_size

        x1_pred = int(x1*ratio_x)
        y1_pred = int(y1*ratio_y)
        x2_pred = int(x2*ratio_x)
        y2_pred = int(y2*ratio_y)

        if flag:
            # print "INFO: Resizing Frames"
            resized = cv2.resize(frame, (hmap_size, hmap_size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255), 2)
            if not os.path.exists('./data/TB-50_OUTPUT/det_img'):
                os.makedirs('./data/TB-50_OUTPUT/det_img')
                os.makedirs('./data/TB-50_OUTPUT/heat_img')
            cv2.imwrite('./data/TB-50_OUTPUT' + '/det_img/' + str(i).zfill(4) + '.jpg', resized)
            cv2.imwrite('./data/TB-50_OUTPUT' + '/heat_img/' + str(i).zfill(4) + '.jpg', 255.0*heat_map)
        else:
            # print "INFO: Scaling Heatmap"
            cv2.rectangle(frame, (x1_pred, y1_pred), (x2_pred, y2_pred), (0,0,255), 2)
            if not os.path.exists('./data/TB-50_OUTPUT/det_img'):
                os.makedirs('./data/TB-50_OUTPUT/det_img')
                os.makedirs('./data/TB-50_OUTPUT/heat_img')
            cv2.imwrite('./data/TB-50_OUTPUT' + '/det_img/' + str(i).zfill(4) + '.jpg', frame)
            cv2.imwrite('./data/TB-50_OUTPUT' + '/heat_img/' + str(i).zfill(4) + '.jpg', 255.0*heat_map)

        x1_true = int(frame_bboxs[i-1][0])
        y1_true = int(frame_bboxs[i-1][1])
        x2_true = int(frame_bboxs[i-1][2]) + int(frame_bboxs[i-1][0])
        y2_true = int(frame_bboxs[i-1][3]) + int(frame_bboxs[i-1][1])

        frames_total += 1
        iou = util.overlap_score([x1_true, y1_true, x2_true, y2_true], [x1_pred, y1_pred, x2_pred, y2_pred])
        if iou>1:
            # print [x1_true, y1_true, x2_true, y2_true], [x1_pred, y1_pred, x2_pred, y2_pred], iou
            iou = 0.0
        score += iou

    print "Score for Directory:", data_dir, score/frames_total


frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)

if sys.argv[2]=='small':
    flag = True
elif sys.argv[2]=='large':
    flag = False

if sys.argv[1]=='simple':
    for (data_dir, frame_bboxs) in zip(data_dirs, frame_bboxs_dirs):
        x_test, y_test = util.get_test_data_simple([data_dir])
        y_predict = model_h.test_simple(x_test, y_test)
        evaluate_dir(y_predict, data_dir, frame_bboxs, flag=flag)
elif sys.argv[1]=='normal':
    for (data_dir, frame_bboxs) in zip(data_dirs, frame_bboxs_dirs):
        x_test_vis, x_test_heat, y_test= util.get_test_data([data_dir])
        y_predict = model_h.test(x_test_vis, x_test_heat, y_test)
        evaluate_dir(y_predict, data_dir, frame_bboxs, flag=flag)
