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

thresh = 0.5
hmSize = 64

def findRectangle(heatMap,thresh):
    minup = hmSize
    minleft = hmSize
    maxright=-1
    maxbottom=-1
    for i in range(hmSize):
        for j in range(hmSize):
            if heatMap[i][j] >= thresh:
                if i < minleft:
                    minleft=i
                if i > maxright:
                    maxright=i
                if j<minup:
                    minup =j
                if j>maxbottom:
                    maxbottom= j
    return minleft,minup,maxright,maxbottom

data_dirs = ['Dancer','Skating1','Girl2']

frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = util.prepare_data(data_dirs)
util.process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs,data_dirs) #save features to file

x_test, y_test= util.get_test_data(data_dirs,[])
y_predict = trainer.test(x_test, y_test)
print y_predict.shape
print y_predict[0]
f1=open('./data/test_output/y_predict.txt', 'wb')
f1.write(y_predict)
f1.close()
path_prefix = './data/TB-50_FEATS/'
i=6
dir_number = 0
current_data_dir = data_dirs[dir_number]
for sample in y_predict:
    i=i+1
    print i,dir_number
    heatMap = sample[5]
    #print heatMap.shape
    heatMap = np.resize(heatMap,(hmSize,hmSize))
    #print heatMap.shape
    frame_path = path_prefix + current_data_dir + '/img/' + str(i).zfill(4) + '.jpg'
    frame = cv2.imread(frame_path)
    width = len(frame[0])
    height = len(frame)
    #resized_image = cv2.resize(frame, (128, 128))
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = findRectangle(heatMap,thresh)
    ratio1 = height/hmSize
    ratio2 = width/hmSize
    top_left_x = top_left_x*ratio1
    top_left_y = top_left_y*ratio2
    bottom_right_x = bottom_right_x*ratio1
    bottom_right_y = bottom_right_y*ratio2
    cv2.rectangle(frame,(top_left_y,top_left_x),(bottom_right_y,bottom_right_x),(0,0,255),1)

    cv2.imwrite(path_prefix + current_data_dir + '/detection_img/' + str(i).zfill(4) + '.jpg', frame)

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
