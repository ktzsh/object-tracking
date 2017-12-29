from ctypes import *
import numpy as np
import os

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class FEATURE(Structure):
    _fields_ = [("size", c_int),
                ("feat", POINTER(c_float))]

class DIMS(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int)]

lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_gpu = lib.load_network_p_gpu
load_net_gpu.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_gpu.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

extract_feat = lib.network_extract_feat
extract_feat.argtypes = [c_void_p, c_int]
extract_feat.restype = FEATURE

layer_dims = lib.layer_dims
layer_dims.argtypes = [c_void_p, c_int]
layer_dims.restype = DIMS


class YOLO:
    net         = None
    meta        = None
    gpu_id      = 1
    cpu_mode    = 0
    NMS         = 0.45
    THRESH      = 0.5
    HIER_THRESH = 0.5

    META        = 'cfg/coco.data'
    CONFIG      = 'cfg/yolo.cfg'
    WEIGHTS     = 'yolo.weights'
    CLASSES     = ['car', 'truck', 'person']

    def __init__(self, argvs=[]):
        self.argv_parser(argvs)
        self.load_detection_model()

    def argv_parser(self, argvs):
        self.cpu_mode = argvs[0]
        self.gpu_id = argvs[1]

    def load_detection_model(self):
        os.chdir('./darknet')
        if self.cpu_mode:
            self.net = load_net(self.CONFIG, self.WEIGHTS, 0)
        else:
            self.net = load_net_gpu(self.CONFIG, self.WEIGHTS, 0, self.gpu_id)
        self.meta = load_meta(self.META)
        os.chdir('../')

    def get_layer_dims(self, n):
        info = layer_dims(self.net, n)
        return (info.w, info.h, info.c)

    def detect(self, image):
        im = load_image(image, 0, 0)
        boxes = make_boxes(self.net)
        probs = make_probs(self.net)
        num =   num_boxes(self.net)
        network_detect(self.net, im, self.THRESH, self.HIER_THRESH, self.NMS, boxes, probs)
        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        free_image(im)
        free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def extract(self, n):
        f = extract_feat(self.net, n)
        feat = np.zeros((f.size))
        for j in range(f.size):
            feat[j] = f.feat[j]
        return feat

    def extract_spatio_info(self, frame_path, layer=24):
        obj_detections = []
        out = self.detect(frame_path)
        vis_feat = self.extract(layer)

        for detection in out:
            if detection[0] in self.CLASSES:
                obj_detections.append(detection)
        return obj_detections, vis_feat
