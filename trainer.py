import os
import json
import importlib
from utility.utils import prepare_data
from models_detection.KerasYOLO import KerasYOLO
from models_tracking.MultiObjDetTracker import MultiObjDetTracker

def single_object_tracking():
    with open("config.json") as config_buffer:
        config = json.loads(config_buffer.read())

    tracker_name  = config["model_tracker"]["name"]
    tracker_class = getattr(importlib.import_module("models_tracking." + tracker_name), tracker_name)
    tracker       = tracker_class()

    tracker.train()

def simult_multi_obj_detection_tracking():
    model = MultiObjDetTracker()
    model.train()

def keras_yolo_obj_detection():
    model = KerasYOLO()
    model.train()

    prefix = 'darknet/data/'
    inputs = ['dog.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg', 'person.jpg']
    model = KerasYOLO()
    for input_instance in inputs:
        model.predict(prefix + input_instance, input_instance)

if __name__=='__main__':

    if not os.path.exists('logs'):
        os.mkdir('logs/')
    if not os.path.exists('models'):
        os.mkdir('models/')

    # single_object_tracking()
    simult_multi_obj_detection_tracking()
