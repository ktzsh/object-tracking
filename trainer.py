import os
from utils.utils import prepare_data
from models_detection.KerasYOLO import KerasYOLO
from models_tracking.TinyTracker import TinyTracker
from models_tracking.TinyHeatmapTracker import TinyHeatmapTracker
from models_tracking.MultiObjDetTracker import MultiObjDetTracker

def single_object_tracking():
    train_data_dirs = [
                    'Human2',     'Human4',   'Human5',     'Human6',   'Human7',
                    'Human9',     'Woman',    'Jogging-1',  'Walking',  'Walking2',
                    'Subway',     'Singer1',  'Walking2',   'Jump',     'Biker',
                    'BlurBody',   'Car2',     'Car4',       'CarDark',  'CarScale',
                    'Suv',        'David3',   'Dancer2',    'Gym',      'Basketball',
                    'Skating2-2'
                ]

    val_data_dirs = [
                        'Human3',  'Human8',  'Jogging-2',  'Skater',     'Girl2',
                        'Dancer',  'Car1',    'Car24',      'Skating2-1'
                    ]

    _CONFIG = {
                '_TRACKER'       : 'TinyTracker', #TinyHeatmapTracker
                '_DETECTOR'      : 'YOLO', #FasterRCNN
                '_CPU_ONLY'      : 0,
                '_TRACKER_GPUID' : 0,
                '_DETECTOR_GPUID': 1,
                '_POOL'          : 'Global', #Max
                '_BATCH_SIZE'    : 32,
                '_MAX_EPOCHS'    : 50
    }

    config = [
                _CONFIG['_DETECTOR'],
                _CONFIG['_CPU_ONLY'],
                _CONFIG['_TRACKER_GPUID'],
                _CONFIG['_DETECTOR_GPUID'],
                _CONFIG['_POOL'],
                _CONFIG['_BATCH_SIZE'],
                _CONFIG['_MAX_EPOCHS']
            ]

    if _CONFIG['_CPU_ONLY']==0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_CONFIG[_TRACKER_GPUID])

    if _CONFIG['_TRACKER']=='TinyTracker':
        tracker = TinyTracker(config)
    elif _CONFIG['_DETECTOR']=='TinyHeatmapTracker':
        tracker = TinyHeatmapTracker(config)


    train_data = prepare_data(train_data_dirs)
    val_data = prepare_data(val_data_dirs)

    tracker.train(train_data, val_data)

if __name__=='__main__':

    # prefix = 'darknet/data/'
    # inputs = ['dog.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg', 'person.jpg']
    # model = KerasYOLO()
    # for input_instance in inputs:
    #     model.predict(prefix + input_instance, input_instance)

    model = MultiObjDetTracker()
    model.train()
