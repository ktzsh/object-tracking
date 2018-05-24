# object-tracking
1. Simultaneous Multiple Object Detection and Tracking System in Keras (Detection network based on YOLOv2 - reimplemented in keras)
2. Single Object Tracking with FasterRCNN and YOLOv2/v3 as detection backends


# Trackers Available:
1. TinyTracker (Static Detection Priors from FasterRCNN or Yolov2/v3)[Single Object]:
2. TinyHeatmapTracker (Static Detection Priors from FasterRCNN or Yolov2/v3)[Single Object]:
3. MultiObjDetTracker (Trainable Detection Priors from Yolov2 reimplmented in Keras)[Multiple Objects]:

NOTE: Yolov2 reimplementation in Keras as standalone detector also available

# Dependencies
Tensorflow
Keras
OpenCV
easydict (for py-faster-rcnn)
cython (for py-faster-rcnn)


#Instllation
1. git clone --recursive https://github.com/kshitiz38/object-tracking.git
    NOTE: If you didn't clone with the --recursive flag run manually the following code
        git submodule update --init --recursive

2a. Darknet
    cd darknet
    Follow instructions at https://pjreddie.com/darknet/install/

2b. Faster RCNN
    cd py-faster-rcnn
    Follow intructions at https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo


# Usage
python trainer.py
Parameters(Open trainer.py):
    1. simult_multi_obj_detection_tracking()
    2. single_object_tracking()
        _TRACKER:  (TinyTracker, TinyHeatmapTracker) [see model archs for details]
        _DETECTOR: (YOLO, FasterRCNN)  [detection priors - more params in each model file]
        _POOL:     (Global, Max, None) [see model archs for details]
        ....

# TODOs
1. Randomize data generator for trackers 1,2
2. Update Usage section README
3. Benchmark simult_multi_obj_detection_tracking for ImagenetVid Challenge


# References
1. https://github.com/Guanghan/ROLO
2. https://github.com/experiencor/keras-yolo2
