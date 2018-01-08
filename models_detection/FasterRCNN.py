import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
caffe_path = os.path.join(this_dir, '..', 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

lib_path = os.path.join(this_dir, '..', 'py-faster-rcnn', 'lib')
add_path(lib_path)

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import caffe
import cv2


class FasterRCNN:
    net         = None
    gpu_id      = 1
    cpu_mode    = 0
    NMS_THRESH  = 0.3
    CONF_THRESH = 0.8

    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')

    def __init__(self, argvs=[]):
        self.argv_parser(argvs)
        self.load_detection_model()

    def argv_parser(self, argvs):
        self.cpu_mode = argvs[0]
        self.gpu_id = argvs[1]

    def load_detection_model(self):
        if not os.path.isfile(self.caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(self.caffemodel))
        if self.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
            cfg.GPU_ID = self.gpu_id

        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        print '\n\nLoaded network', self.caffemodel

    def extract_spatio_info(self, frame_path, layer='fc7'):
        im = cv2.imread(frame_path)
        scores, boxes = im_detect(self.net, im)

        vis_feat = self.net.blobs[layer].data[0]
        print vis_feat

        obj_detections = []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            obj_detections += dets
            print dets

        return obj_detections, vis_feat
