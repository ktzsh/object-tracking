import numpy as np
import os
import xml.etree.ElementTree as ET
import copy
import cv2

def prepare_data(data_dirs):
    path_prefix = 'data/VisualTB/'
    frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs = [], [], []
    print "All Directories List:", data_dirs

    for data_dir in data_dirs:
        frame_paths, frame_bboxs, frame_width, frame_height = [], [], 0, 0
        print "Searching Data in Directory:", data_dir.split('-')[0]

        groundtruth_path = path_prefix + data_dir + '/groundtruth_rect.txt'
        if data_dir=='Jogging-1':
            groundtruth_path = path_prefix + 'Jogging' + '/groundtruth_rect.1.txt'
        if data_dir=='Jogging-2':
            groundtruth_path = path_prefix + 'Jogging' + '/groundtruth_rect.2.txt'
        if data_dir=='Human4-1':
            groundtruth_path = path_prefix + 'Human4' + '/groundtruth_rect.1.txt'
        if data_dir=='Skating2-1':
            groundtruth_path = path_prefix + 'Skating2' + '/groundtruth_rect.1.txt'
        if data_dir=='Skating2-2':
            groundtruth_path = path_prefix + 'Skating2' + '/groundtruth_rect.2.txt'

        with open(groundtruth_path, 'rb') as f_handle:
            lines = f_handle.readlines()
            for i,line in enumerate(lines):
                frame_path = path_prefix + data_dir.split('-')[0] + '/img/' + str(i+1).zfill(4) + '.jpg'
                if i==0:
                    frame = cv2.imread(frame_path)
                    frame_height, frame_width, frame_channel = frame.shape[0], frame.shape[1], frame.shape[2]

                if data_dir=='Jogging-1' or data_dir=='Jogging-2' or data_dir=='Woman' or data_dir=='Walking' or data_dir=='Walking2' or data_dir=='Subway' or data_dir=='Singer1' or data_dir=='Girl' or data_dir=='BlurBody' or data_dir=='Car4' or data_dir=='CarScale' or data_dir=='Skating2-1' or data_dir=='Skating2-2' :
                    frame_bbox = line.rstrip('\n').split()
                else:
                    frame_bbox = line.rstrip('\n').split(',')

                frame_bbox = [float(i) for i in frame_bbox]
                frame_bboxs.append(frame_bbox)
                frame_paths.append(frame_path)

        print "Number of samples:", len(frame_paths)
        frame_paths_dirs.append(frame_paths)
        frame_bboxs_dirs.append(frame_bboxs)
        frame_dim_dirs.append([frame_height, frame_width, frame_channel])

    return frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs


def generate_heatmap_feat(det_x, det_y, det_w, det_h, hmap_size=32):
    heatmap = np.zeros((hmap_size, hmap_size))
    scaled_x, scaled_y, scaled_h, scaled_w = int(det_x*hmap_size), int(det_y*hmap_size), int(det_h*hmap_size), int(det_w*hmap_size)
    heatmap[scaled_y:(scaled_y+scaled_h+1), scaled_x:(scaled_x+scaled_w+1)] = 1.0
    heatmap_feat = heatmap.reshape((-1))
    return heatmap_feat


def generate_rectangle_from_heatmap(heat_map, thresh=0.75, hmap_size=32):
    x1 = hmap_size
    y1 = hmap_size
    y2 = -1
    x2 = -1

    for i in range(hmap_size):
        for j in range(hmap_size):
            if heat_map[i][j] >= thresh:
                if i < y1:
                    y1 = i
                if i > y2:
                    y2 = i
                if j < x1:
                    x1 = j
                if j > x2:
                    x2 = j

    return x1, y1, x2, y2


def overlap_score(y_true, y_pred):

    x1_true = y_true[0]
    y1_true = y_true[1]
    x2_true = y_true[2]
    y2_true = y_true[3]

    x1_pred = y_pred[0]
    y1_pred = y_pred[1]
    x2_pred = y_pred[2]
    y2_pred = y_pred[3]

    x1 = max(x1_true, x1_pred)
    y1 = max(y1_true, y1_pred)
    x2 = min(x2_true, x2_pred)
    y2 = min(y2_true, y2_pred)
    intersection = float(abs((x1 - x2) * (y1 - y2)))
    union = float(abs((x1_true - x2_true) * (y1_true - y2_true))) + float(abs((x1_pred - x2_pred) * (y1_pred - y2_pred))) - intersection

    return (intersection/union)

def average_overlap_score(y_true, y_pred):

    score = 0.0
    total = 0
    for i, (y_true_sample, y_pred_sample) in enumerate(zip(y_true, y_pred)):
        score += overlap_score(y_true_sample, y_pred_sample)
        total = i
    return score/(total+1)


class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h

        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

def normalize(image):
    image = image / 255.

    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2

    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def draw_boxes(image, boxes, labels):

    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0,255,0), 2)

    return image

def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]

                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]

                    box = BoundBox(x, y, w, h, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
