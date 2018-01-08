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

        self.lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)

        self.make_boxes          = self.lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype  = POINTER(BOX)

        self.free_ptrs          = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes          = self.lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype  = c_int

        self.make_probs          = self.lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype  = POINTER(POINTER(c_float))

        self.load_net          = self.lib.load_network_p
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype  = c_void_p

        self.load_net_gpu          = self.lib.load_network_p_gpu
        self.load_net_gpu.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_gpu.restype  = c_void_p

        self.free_image          = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.load_meta          = self.lib.get_metadata
        self.load_meta.argtypes = [c_char_p]
        self.load_meta.restype  = METADATA

        self.load_image          = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype  = IMAGE

        self.network_detect          = self.lib.network_detect
        self.network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        self.extract_feat          = self.lib.network_extract_feat
        self.extract_feat.argtypes = [c_void_p, c_int]
        self.extract_feat.restype  = FEATURE

        self.layer_dims          = self.lib.layer_dims
        self.layer_dims.argtypes = [c_void_p, c_int]
        self.layer_dims.restype  = DIMS

        self.argv_parser(argvs)
        self.load_detection_model()

    def argv_parser(self, argvs):
        self.cpu_mode = argvs[0]
        self.gpu_id = argvs[1]

    def load_detection_model(self):
        os.chdir('./darknet')
        if self.cpu_mode:
            self.net = self.load_net(self.CONFIG, self.WEIGHTS, 0)
        else:
            self.net = self.load_net_gpu(self.CONFIG, self.WEIGHTS, 0, self.gpu_id)
        self.meta = self.load_meta(self.META)
        os.chdir('../')

    def get_layer_dims(self, n):
        info = layer_dims(self.net, n)
        return (info.w, info.h, info.c)

    def detect(self, image):
        print "DEBUG 4", image
        im = self.load_image(image, 0, 0)
        print "DEBUG 5"
        boxes = self.make_boxes(self.net)
        print "DEBUG 6"
        probs = self.make_probs(self.net)
        print "DEBUG 7"
        num =   self.num_boxes(self.net)
        print "DEBUG 8"
        self.network_detect(self.net, im, self.THRESH, self.HIER_THRESH, self.NMS, boxes, probs)
        print "DEBUG 9"
        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def extract(self, n):
        f = self.extract_feat(self.net, n)
        feat = np.zeros((f.size))
        for j in range(f.size):
            feat[j] = f.feat[j]
        return feat

    def extract_spatio_info(self, frame_path, layer=24):
        obj_detections = []
        print "DEBUG 1", frame_path
        out = self.detect(frame_path)
        print "DEBUG 2"
        vis_feat = self.extract(layer)
        print "DEBUG 3"
        for detection in out:
            if detection[0] in self.CLASSES:
                obj_detections.append(detection)
        return obj_detections, vis_feat
