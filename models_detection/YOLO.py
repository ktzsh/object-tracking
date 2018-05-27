import os
import json
import numpy as np
from ctypes import *

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

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
    def __init__(self, argvs=[]):

        with open("config.json") as config_buffer:
            self.config = json.loads(config_buffer.read())

        self.gpu_id      = self.config["train"]["dgpu_id"]
        self.cpu_mode    = self.config["train"]["cpu_only"]

        self.CLASSES     = [s.lower() for s in self.config["train"]["classes"]]

        self.NMS         = self.config["model_detector"]["nms"]
        self.THRESH      = self.config["model_detector"]["thresh"]
        self.HIER_THRESH = self.config["model_detector"]["hier_thresh"]

        self.META        = self.config["model_detector"]["meta_file"]
        self.CONFIG      = self.config["model_detector"]["config_file"]
        self.WEIGHTS     = self.config["model_detector"]["weights_file"]

        lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)

        lib.network_width.argtypes  = [c_void_p]
        lib.network_width.restype   = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype  = c_int

        self.set_gpu          = lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.get_network_boxes          = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype  = POINTER(DETECTION)

        self.free_detections          = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs          = lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.load_net          = lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype  = c_void_p

        self.do_nms_obj          = lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image          = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.load_meta          = lib.get_metadata
        self.load_meta.argtypes = [c_char_p]
        self.load_meta.restype  = METADATA
        # self.lib.get_metadata.argtypes = [c_char_p]
        # self.lib.get_metadata.restype  = METADATA

        self.load_image          = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype  = IMAGE

        self.rgbgr_image          = lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image          = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype  = POINTER(c_float)

        # self.load_net          = self.lib.load_network_p
        # self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        # self.load_net.restype  = c_void_p
        #
        # self.load_net_gpu          = self.lib.load_network_p_gpu
        # self.load_net_gpu.argtypes = [c_char_p, c_char_p, c_int, c_int]
        # self.load_net_gpu.restype  = c_void_p

        self.extract_feat          = lib.network_extract_feat
        self.extract_feat.argtypes = [c_void_p, c_int]
        self.extract_feat.restype  = FEATURE

        self.layer_dims          = lib.layer_dims
        self.layer_dims.argtypes = [c_void_p, c_int]
        self.layer_dims.restype  = DIMS

        self.argv_parser(argvs)
        self.load_detection_model()

    def argv_parser(self, argvs):
        self.cpu_mode = argvs[0]
        self.gpu_id   = argvs[1]

    def load_detection_model(self):
        os.chdir('./darknet')
        if not self.cpu_mode:
            self.set_gpu(self.gpu_id)
        self.net  = self.load_net(self.CONFIG, self.WEIGHTS, 0)
        self.meta = self.load_meta(self.META)
        os.chdir('../')

    def get_layer_dims(self, n):
        info = self.layer_dims(self.net, n)
        return (info.h, info.w, info.c)

    def detect(self, image):
        im   = self.load_image(image, 0, 0)
        num  = c_int(0)
        pnum = pointer(num)

        self.predict_image(self.net, im)

        dets = self.get_network_boxes(self.net, im.w, im.h, self.THRESH, self.HIER_THRESH, None, 0, pnum)
        num  = pnum[0]

        if (self.NMS): self.do_nms_obj(dets, num, self.meta.classes, self.NMS);

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    # midpoint coordinates and height width
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def extract(self, n):
        f    = self.extract_feat(self.net, n)
        feat = np.zeros((f.size))

        for j in range(f.size):
            feat[j] = f.feat[j]
        return feat

    def extract_spatio_info(self, frame_path, layer=24):
        obj_detections = []
        out            = self.detect(frame_path)
        vis_feat       = self.extract(layer)

        for detection in out:
            if detection[0] in self.CLASSES:
                obj_detections.append(detection)
        return obj_detections, vis_feat
