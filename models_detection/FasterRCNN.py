import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
caffe_path = os.path.join(this_dir, '..', 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
lib_path = osp.join(this_dir, '..', 'py-faster-rcnn', 'lib')
add_path(caffe_path)
add_path(lib_path)

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe
import cv2


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def load_detection_model(cpu_mode=1, gpu_id=0):
    prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        cfg.GPU_ID = gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network', caffemodel
    return net

def extract_spatio_info(net, frame_path, layer_name='fc7'):
    im = cv2.imread(frame_path)
    scores, boxes = im_detect(net, im)

    feat = net.blobs[layer_name].data[0]
    print feat

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print dets
