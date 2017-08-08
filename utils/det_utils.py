import numpy as np
from ctypes import *
import sys
import os
import cv2

lib = CDLL('/root/workspace/PedestrainDetection/darknet/libdarknet.so', RTLD_GLOBAL)

load_net = lib.load_detector
load_net.argtypes = [c_char_p, c_char_p, c_char_p]
load_net.restype = c_void_p

detect_obj = lib.pyrun_detector
detect_obj.argtypes = [c_void_p, c_char_p, c_float, c_float, c_char_p]
detect_obj.restype = py_object

extract_feat = lib.pyfeat_extract
extract_feat.argtypes = [c_int, c_void_p]
extract_feat.restype = py_object

os.chdir('./darknet')
global net
net = load_net("cfg/coco.data", "cfg/yolo.cfg", "yolo.weights")
global net_dim
net_dim = 608

def prepare_data(data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1', 'Jogging-2','Walking','Walking2']):
    path_prefix = '/root/Artifacia-Data/TB-50/'
    frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = [], [], []
    print "All Directories List:", data_dirs

    for data_dir in data_dirs:
        frame_paths, frame_bboxs, frame_width, frame_height = [], [], 0, 0
        print "Reading Directory:", data_dir.split('-')[0]

        groundtruth_path = path_prefix + data_dir + '/groundtruth_rect.txt'
        if data_dir=='Jogging-1':
            groundtruth_path = path_prefix + 'Jogging' + '/groundtruth_rect.1.txt'
        if data_dir=='Jogging-2':
            groundtruth_path = path_prefix + 'Jogging' + '/groundtruth_rect.2.txt'

        with open(groundtruth_path, 'rb') as f_handle:
            lines = f_handle.readlines()
            for i,line in enumerate(lines):
                frame_path = path_prefix + data_dir.split('-')[0] + '/img/' + str(i+1).zfill(4) + '.jpg'
                if i==0:
                    frame = cv2.imread(frame_path)
                    frame_height, frame_width, frame_channel = frame.shape[0], frame.shape[1], frame.shape[2]

                if data_dir=='Jogging-1' or data_dir=='Jogging-2' or data_dir=='Woman' or data_dir=='Walking' or data_dir=='Walking2':
                    frame_bbox = line.rstrip('\n').split()
                    frame_bboxs.append(frame_bbox)
                else:
                    frame_bbox = line.rstrip('\n').split(',')
                    frame_bboxs.append(frame_bbox)

                frame_paths.append(frame_path)

        print "Number of samples:", len(frame_paths)
        print "Done.."
        frame_paths_dirs.append(frame_paths)
        frame_bboxs_dirs.append(frame_bboxs)
        frame_dim_dirs.append([frame_height, frame_width, frame_channel])
    return frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs

def extract_spatio_info(frame_path):
    obj_detections, vis_feat = [], None
    out = detect_obj(net, frame_path, 0.24, 0.5, "prediction")
    vis_feat = extract_feat(30, net)
    for detection in out:
        if detection.get('class')=='car' or detection.get('class')=='person':
            obj_detections.append(detection)
    return obj_detections, vis_feat


def generate_heatmap_feat(det_x, det_y, det_h, det_w, smap=32):

    heatmap = np.zeros((smap, smap))
    scaled_x, scaled_y, scaled_h, scaled_w = int(det_x*smap), int(det_y*smap), int(det_h*smap), int(det_w*smap)
    heatmap[scaled_y:(scaled_y+scaled_h+1), scaled_x:(scaled_x+scaled_w+1)] = 1.0
    heatmap_feat = heatmap.reshape((-1))
    return heatmap_feat


def process_data(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs, data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2']):
    for i, (frame_paths, frame_bboxs, frame_dim) in enumerate(zip(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs)):
        frame_height, frame_width = frame_dim[0], frame_dim[1]

        vis_data_file = '../data/vis_feats_' + data_dirs[i]
        heatmap_data_file = '../data/heatmap_feats_' + data_dirs[i]

        # if i<0:
        #     continue

        print "Extracting features for Data in Directory:", data_dirs[i].split('-')[0]
        for j, (frame_path, frame_bbox) in enumerate(zip(frame_paths,frame_bboxs)):

            # if j<0:
            #     continue

            _, feat = extract_spatio_info(frame_path)

            tmp = np.array(feat, dtype='float32')
            tmp = tmp.reshape((19,19,1024))
            feat = np.amax(tmp, axis=(0,1))

            det_x = float(frame_bbox[0])/frame_width
            det_y = float(frame_bbox[1])/frame_height
            det_h = float(frame_bbox[3])/frame_height
            det_w = float(frame_bbox[2])/frame_width

            heatmap_feat = generate_heatmap_feat(det_x, det_y, det_h, det_w)
            np.savetxt(f_vis, (feat.reshape((1,-1))), delimiter=',')
            np.savetxt(f_heat, (heatmap_feat.reshape((1,-1))), delimiter=',')


def read_data(sequence_length=6, vis_feat_size=1024, heatmap_feat_size=1024, data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2']):
    for data_dir in data_dirs:

        print "Reading Directory:", data_dir
        vis_data_file = '../data/vis_feats_' + data_dir
        heatmap_data_file = '../data/heatmap_feats_' + data_dir

        vis_feat = np.genfromtxt(vis_data_file, dtype='float32', delimiter=',')
        heatmap_feat = np.genfromtxt(heatmap_data_file, dtype='float32', delimiter=',')

        leng = vis_feat.shape[0]
        x = np.zeros((leng - sequence_length, sequence_length, (vis_feat_size + heatmap_feat_size)))
        y = np.zeros((leng - sequence_length, sequence_length, heatmap_feat_size))

        for i in range(leng-sequence_length):

            x_sample = np.zeros((sequence_length, (vis_feat_size + heatmap_feat_size)))
            y_sample = np.zeros((sequence_length, heatmap_feat_size))
            for j in range(sequence_length):
                x_sample[j, :] = np.append(vis_feat_size[i+j], heatmap_feat[i+j])
                y_sample[j, :] = heatmap_feat[i+j+1]

            x[i] = x_sample
            y[i] = y_sample

        yield (x,y)

def get_trainval_data(data_dirs = ['Human2','Human3','Human4','Human5','Human6','Human7','Human8','Human9','Woman','Jogging-1','Jogging-2','Walking','Walking2'], val_data_dirs=['Human3','Human8','Jogging-2','Walking2']):

    x_train = None
    y_train = None
    x_val = None
    y_val = None

    val_index = [data_dirs.index(data_dir) for data_dir in val_data_dirs]
    data_generator = read_data()
    for i,(x,y) in enumerate(data_generator):

        if i in val_index:
            if x_val==None and y_val==None:
                x_val = x
                y_val = y
            else:
                x_val = np.append(x_val, x, axis=0)
                y_val = np.append(y_val, y, axis=0)
        else:
            if x_train==None and y_train==None:
                x_train = x
                y_train = y
            else:
                x_train = np.append(x_train, x, axis=0)
                y_train = np.append(y_train, y, axis=0)

    print "Shapes of Train/Val X/Y Data:", x_train.shape, y_train.shape, x_val.shape, y_val.shape
    return x_train, y_train, x_val, y_val
