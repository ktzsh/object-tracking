import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from models_detection.YOLO import YOLO
from models_detection.FasterRCNN import FasterRCNN
#  --------------------------------------------------------------------------------------------------------------------

class BaseTracker(object):

    model_name        = 'BaseTracker'
    log_path          = 'logs/' + model_name + '_'
    all_models_path   = 'weights/MODEL_' + model_name + '_'
    best_weights_path = 'weights/WEIGHTS_' + model_name + '_'

    model_tracker      = None
    model_detector     = None
    detection_model    = None

    def __init__(self, argv):
        self.argv_parser(argv)
        self.load_detection_model()

    def argv_parser(self, argv):
        self.detection_model = argv[0]
        self.cpu_mode = argv[1]
        self.tgpu_id = argv[2]
        self.dgpu_id = argv[3]
        self.pool = argv[4]
        self.batch_size = argv[5]
        self.max_epochs = argv[6]
        self.detection_fv_layer = argv[7]
        if self.tgpu_id==self.dgpu_id:
            print '#####################################################################################'
            print '# [WARNING] Models for Detection and Tracking should be loaded on DIFFERENT GPUs OR #'
            print '# save the extracted features from detection network first then run the tracker.    #'
            print '#####################################################################################'


    def load_detection_model(self):
        if self.detection_model=='YOLO':
            self.model_detector = YOLO([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu\
            self._w, self._h, self._c = self.model_detector.get_layer_dims(self.detection_fv_layer)

        elif self.detection_model=='FasterRCNN':
            self.model_detector = FasterRCNN([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu
            self.detection_fv_layer = 'relu5_3'
            self._w = 1 #CORRECT THIS
            self._h = 1 #CORRECT THIS
            self._c = 512


    def load_tracker_model(self):
        raise NotImplementedError


    def get_batch(self, frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs):
        raise NotImplementedError


    def train(self, train_data_dirs, val_data_dirs):
        t_frame_paths_dirs, t_frame_bboxs_dirs, t_frame_dim_dirs = train_data_dirs
        v_frame_paths_dirs, v_frame_bboxs_dirs, v_frame_dim_dirs = val_data_dirs

        t_data_size = sum( len(t) - self.SEQUENCE_LENGTH + 1 for t in t_frame_paths_dirs)
        v_data_size = sum( len(v) - self.SEQUENCE_LENGTH + 1 for v in v_frame_paths_dirs)

        model_checkpointer = ModelCheckpoint(filepath=self.all_models_path + self.detection_model + '.h5',
                                        monitor='val_loss',
                                        mode='min',
                                        save_weights_only=False,
                                        save_best_only=False,
                                        verbose=0)

        weights_checkpointer = ModelCheckpoint(filepath=self.best_weights_path + self.detection_model + '.h5',
                                        monitor='val_loss',
                                        mode='min',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        verbose=2)

        earlystopper = EarlyStopping(monitor='val_loss',
                                    mode='min',
                                    patience=10,
                                    verbose=2)

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=5,
                                            verbose=2,
                                            epsilon=1e-4,
                                            mode='min')

        csv_logger = CSVLogger(self.log_path + self.detection_model + '.log', append=False)

        self.model_tracker.fit_generator(
                            self.get_batch(t_frame_paths_dirs, t_frame_bboxs_dirs, t_frame_dim_dirs),
                            steps_per_epoch=t_data_size//self.batch_size,
                            epochs=self.max_epochs,
                            callbacks=[earlystopper, model_checkpointer, weights_checkpointer, csv_logger, reduce_lr_loss],
                            validation_data=self.get_batch(v_frame_dim_dirs, v_frame_paths_dirs, v_frame_bboxs_dirs),
                            validation_steps=v_data_size//self.batch_size
                            )
