from ctypes import *
import os

lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)

run_demo = lib.demo
run_demo.argtypes = [c_char_p, c_char_p, c_float, c_int, c_char_p, POINTER(c_char_p), c_int, c_int, c_char_p, c_int, c_float, c_int, c_int, c_int, c_int]
run_demo.restype = c_void_p

read_data_cfg = lib.read_data_cfg
read_data_cfg.argtypes = [c_char_p]
read_data_cfg.restype = c_void_p

option_find_int = lib.option_find_int
option_find_int.argtypes = [c_void_p, c_char_p, c_int]
option_find_int.restype = c_int

option_find_str = lib.option_find_str
option_find_str.argtypes = [c_void_p, c_char_p, c_char_p]
option_find_str.restype = c_char_p

get_labels = lib.get_labels
get_labels.argtypes = [c_char_p]
get_labels.restype = POINTER(c_char_p)


def run_yolo_sequence():

    os.chdir('darknet')
    options = read_data_cfg('cfg/coco.data')
    classes = option_find_int(options, 'classes', 20)
    name_list = option_find_str(options, 'names', 'data/names.list')
    names = get_labels(name_list)
    run_demo('cfg/yolo.cfg', 'yolo.weights', 0.24, 0, '/root/workspace/PedestrainDetection/demo.mp4', names, classes, 0, '../data/video/demo', 3, 0.5, 0, 0, 0, 0)
    os.chdir('../')
