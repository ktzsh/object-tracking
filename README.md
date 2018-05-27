# Object Tracking
1. Simultaneous Multiple Object Detection and Tracking System in Keras (Detection network based on YOLOv2 - reimplemented in keras)
2. Single Object Tracking with FasterRCNN and YOLOv2/v3 as detection backends

# Trackers Available:
1. TinyTracker (Static Detection Priors from FasterRCNN or Yolov2/v3)[Single Object]:
2. TinyHeatmapTracker (Static Detection Priors from FasterRCNN or Yolov2/v3)[Single Object]:
3. MultiObjDetTracker (Trainable Detection Priors from Yolov2 reimplmented in Keras)[Multiple Objects]:

NOTE: Yolov2 reimplementation in Keras as standalone detector also available

## Dependencies
1. Tensorflow
2. Keras
3. OpenCV
4. easydict (for py-faster-rcnn)
5. cython (for py-faster-rcnn)
6. imgaug

## Instllation
1. Run `git clone --recursive https://github.com/kshitiz38/object-tracking.git`
    - NOTE: If you didn't clone with the --recursive flag run manually the following code
        `git submodule update --init --recursive`

2. Darknet
    - Follow instructions at https://pjreddie.com/darknet/install/
        `cd darknet && make`
    - NOTE: I recommend disabling CUDNN in Makefile since it gives strange results depending on your version

3. Faster RCNN
    - Follow intructions at https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo
        ````
        cd py-faster-rcnn
        cd lib && make && cd ../
        cd caffe-fast-rcnn && mkdir build && cd build && cmake ..
        make all && make install
        ````

## Usage
1. For Single Object Tracking
    1. Modify Parameters in config.jon
    2. Convert Datasets to PASCAL VOC format if not already
        - Run `python utility/tb_to_pascal.py' or 'python utility/tb_to_pascal.py` or write one for your own dataset
    3. Run `python trainer.py`
2. For Simultaneous Multiple Object Detection and Tracking
    1. Modify Parameters in `KerasYOLO.py` and `MultiObjDetTracker.py`
    2. Convert Datasets like above specify paths in `MultiObjDetTracker.py` already done for ImageNet Vid and MOT17
    3. Run `python trainer.py`

### NOTE :
- Call `single_object_tracking()` in `trainer.py` for Single Object Detection with fixed detection priors from Other Detection backends
- Call `simult_multi_obj_detection_tracking()` in `trainer.py` for Simultaneous Multiple Object Detction and Tracking with Yolov2 Reimplemented in Keras

## Model Architectures
- Coming Soon!!
- Feel free to figure out yourself!! See `models_tracking` and `models_detection` directories

## TODOs
- [ ] Add theory and model architectures explaination
- [ ] Add config.json file for parameters for MultiObjDetTracker and KerasYOLO
- [ ] Benchmark for ImagenetVid Challenge, MOT and VisualTB Datasets
- [ ] Add support for Detectron models as detection backend

## References
1. https://github.com/Guanghan/ROLO
2. https://github.com/experiencor/keras-yolo2
