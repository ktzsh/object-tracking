import numpy as np

def extract_spatio_info(frame):


def generate_heatmap(frame_height, frame_width, det_x, det_y, det_h, det_w, smap=32):

    heatmap = np.zeros((smap, smap))
    scaled_x, scaled_y, scaled_h, scaled_w = det_x*smap/frame_width, det_y*smap/frame_height, det_h*smap/frame_height, det_w*smap/frame_width
    heatmap[scaled_y:(scaled_y+scaled_h+1), scaled_x:(scaled_x+scaled_w+1)] = 1.0
    return heatmap

def generate_heatmap_feat(frame, detection):

    frame_height =
    frame_width =
    det_x, det_y, det_h, det_w = detection[0], detection[1], detection[2], detection[3]
    heatmap = generate_heatmap(frame_height, frame_width, det_x, det_y, det_h, det_w)
    heatmap_feat = heatmap.reshape((-1))
    return heatmap_feat

def process_frame(frame):

    # obj_detections -> [x, y, h, w], vis_feat -> (4096)
    obj_detections, vis_feat = extract_spatio_info(frame)
