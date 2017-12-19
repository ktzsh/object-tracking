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


def load_detection_model():
    os.chdir('./darknet')
    net = load_net("cfg/yolo.cfg", "yolo.weights", 0)
    meta = load_meta("cfg/coco.data")
    os.chdir('../')
    return net, meta

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

def extract(net, n):
    f = extract_feat(net, n)
    feat = np.zeros((f.size))
    for j in range(f.size):
        feat[j] = f.feat[j]
    return feat

def extract_spatio_info(net, meta, frame_path, n):
    obj_detections = []
    out = detect(net, meta, frame_path)
    vis_feat = extract(net, n)

    for detection in out:
        if detection[0]=='car' or detection[0]=='truck' or detection[0]=='person':
            obj_detections.append(detection)
    return obj_detections, vis_feat

def get_layer_dims(net, n):
    info = layer_dims(net, n)
    return (info.w, info.h, info.c)
