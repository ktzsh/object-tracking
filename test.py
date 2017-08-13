import utils.det_utils as util
import scripts.train as trainer
#from resizeimage import resizeimage
import cv2
import numpy as np
import math
import random
import time
import sys
import os

def findRectangle(heatMap,thresh):
    minup = 32
    minleft=32
    maxright=-1
    maxbottom=-1
    for i in range(32):
        for j in range(32):
            if heatMap[i][j] >= thresh:
                if i < minleft:
                    minleft=i
                if i > maxright:
                    maxright=i
                if j<minup:
                    minup =j
                if j>maxbottom:
                    maxbottom= j
    return 4*minleft,4*minup,4*maxright,4*maxbottom

data_dirs = ['Dancer','Skating1','Girl2']


thresh = 0.5
#frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)
#util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs,data_dirs) #save features to file
x_test, y_test= util.get_test_data(data_dirs,[])
y_predict = trainer.test(x_test, y_test)
print y_predict.shape
print y_predict[0]
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
    heatMap = sample[5]
    #print heatMap.shape
    heatMap = np.resize(heatMap,(32,32))
    #print heatMap.shape
    frame_path = path_prefix + current_data_dir + '/img/' + str(i).zfill(4) + '.jpg'
    frame = cv2.imread(frame_path)
    resized_image = cv2.resize(frame, (128, 128))
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = findRectangle(heatMap,thresh)
    cv2.rectangle(resized_image,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),1)

    cv2.imwrite(path_prefix + current_data_dir + '/detection_img/' + str(i).zfill(4) + '.jpg', resized_image)

    cv2.imwrite(path_prefix + current_data_dir + '/heatmap_img/' + str(i).zfill(4) + '.jpg', 255*heatMap)


    if dir_number is 0 and i is 225:
        dir_number = 1
        i = 6
        print current_data_dir +"Done"
        current_data_dir = data_dirs[dir_number]
    print i,dir_number
    if dir_number == 1 and i == 400:
        dir_number = 2
        i = 6
        print current_data_dir +"Done"
        current_data_dir = data_dirs[dir_number]
