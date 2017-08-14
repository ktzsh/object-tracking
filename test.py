import utils.det_utils as util
import scripts.model as model_h

import cv2
import numpy as np
import math
import random
import time
import sys
import os

x_test, y_test, x_test_vis, x_test_heat, y_test = None, None, None, None, None

thresh = 0.5
hmap_size = 64

def find_rectangle(heat_map, thresh):
    min_up = hmap_size
    min_left = hmap_size
    max_right=-1
    max_bottom=-1
    for i in range(hmap_size):
        for j in range(hmap_size):
            if heat_map[i][j] >= thresh:
                if i < min_left:
                    min_left=i
                if i > max_right:
                    max_right=i
                if j<min_up:
                    min_up =j
                if j>max_bottom:
                    max_bottom= j
    return min_left,min_up,max_right,max_bottom

data_dirs = ['Human8']

if sys.argv[1]=='simple':
    x_test, y_test = util.get_test_data_simple(data_dirs)
    y_predict = model_h.test_simple(x_test, y_test)
elif sys.argv[1]=='normal':
    x_test_vis, x_test_heat, y_test= util.get_test_data(data_dirs)
    y_predict = model_h.test(x_test_vis, x_test_heat, y_test)


f1=open('./data/test_output/y_predict.txt', 'wb')
f1.write(y_predict)
f1.close()

path_prefix = './data/TB-50/'

i=6
dir_number = 0
current_data_dir = data_dirs[dir_number]
for sample in y_predict:
    i=i+1
    print i,dir_number
    heat_map = sample[5]
    #print heat_map.shape
    heat_map = np.resize(heat_map,(hmap_size, hmap_size))
    #print heat_map.shape
    frame_path = path_prefix + current_data_dir + '/img/' + str(i).zfill(4) + '.jpg'
    frame = cv2.imread(frame_path)
    height, width, channels = frame.shape
    #width = len(frame[0])
    #height = len(frame)
    #resized_image = cv2.resize(frame, (128, 128))
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = find_rectangle(heat_map,thresh)
    ratio1 = height/hmap_size
    ratio2 = width/hmap_size
    top_left_x = top_left_x*ratio1
    top_left_y = top_left_y*ratio2
    bottom_right_x = bottom_right_x*ratio1
    bottom_right_y = bottom_right_y*ratio2
    cv2.rectangle(frame,(top_left_y,top_left_x),(bottom_right_y,bottom_right_x),(0,0,255),1)

    cv2.imwrite(path_prefix + current_data_dir + '/detection_img/' + str(i).zfill(4) + '.jpg', frame)
    cv2.imwrite(path_prefix + current_data_dir + '/heat_map_img/' + str(i).zfill(4) + '.jpg', 255*heat_map)

'''
    if dir_number is 0 and i is 1500:
        dir_number = 1
        i = 6
        print current_data_dir +"Done"
        current_data_dir = data_dirs[dir_number]
    print i,dir_number
    if dir_number == 1 and i == 225:
        dir_number = 2
        i = 6
        print current_data_dir +"Done"
        current_data_dir = data_dirs[dir_number]
'''
