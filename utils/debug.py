import cv2
import os

def draw(data_dirs, frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs):

    hmap_size = 64

    for i, (frame_paths, frame_bboxs, frame_dim) in enumerate(zip(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs)):
        frame_height, frame_width = frame_dim[0], frame_dim[1]

        print "DEBUG: Directory", data_dirs[i]
        for j, (frame_path, frame_bbox) in enumerate(zip(frame_paths,frame_bboxs)):

            if j>= 20:
                break

            det_x = float(frame_bbox[0])/frame_width
            det_y = float(frame_bbox[1])/frame_height
            det_h = float(frame_bbox[3])/frame_height
            det_w = float(frame_bbox[2])/frame_width

            det_x = int(det_x*hmap_size)
            det_y = int(det_y*hmap_size)
            det_h = int(det_h*hmap_size)
            det_w = int(det_w*hmap_size)

            frame = cv2.imread(frame_path)
            resized = cv2.resize(frame, (hmap_size, hmap_size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (det_x, det_y),(det_x + det_w, det_y + det_h), (0,0,255), 2)
            if not os.path.exists('./data/debug/' + data_dirs[i]):
                os.makedirs('./data/debug/' + data_dirs[i])
            cv2.imwrite('./data/debug/' + data_dirs[i] + '/' + str(j).zfill(4) + '.jpg', resized)
