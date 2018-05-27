import os
import json
import numpy as np
import imgaug as ia
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard

from models_detection.YOLO import YOLO
from models_detection.FasterRCNN import FasterRCNN
#  --------------------------------------------------------------------------------------------------------------------
class BaseTracker(object):
    def __init__(self):
        with open("config.json") as config_buffer:
            self.config = json.loads(config_buffer.read())

        self.detection_model    = self.config["model_detector"]["name"]
        self.detection_fv_layer = self.config["model_detector"]["fv_layer"]

        self.cpu_mode           = self.config["train"]["cpu_only"]
        self.tgpu_id            = self.config["train"]["tgpu_id"]
        self.dgpu_id            = self.config["train"]["dgpu_id"]

        self.pool               = self.config["train"]["pool"]
        self.batch_size         = self.config["train"]["batch_size"]
        self.max_epochs         = self.config["train"]["max_epochs"]
        self.sequence_length    = self.config["model_tracker"]["sequence_length"]

        self.classes            = self.config["train"]["classes"]

        self.model_name         = self.config["model_tracker"]["name"]
        self.tensorboard_dir    = self.config["train"]["tensorboard_dir"]
        self.saved_model_path   = self.config["train"]["saved_model_dir"] + self.model_name

        self.train_image_folder = self.config["train"]["train_image_folder"]
        self.train_annot_folder = self.config["train"]["train_annot_folder"]
        self.val_image_folder   = self.config["val"]["val_image_folder"]
        self.val_annot_folder   = self.config["val"]["val_annot_folder"]

        if not self.cpu_mode:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.tgpu_id)

        if self.tgpu_id==self.dgpu_id:
            print '#####################################################################################'
            print '# [WARNING] Models for Detection and Tracking should be loaded on DIFFERENT GPUs OR #'
            print '# save the extracted features from detection network first then run the tracker.    #'
            print '#####################################################################################'

        self.load_detection_model()


    def load_detection_model(self):
        if self.detection_model=='YOLO':
            self.model_detector = YOLO([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu\
            self._w, self._h, self._c = self.model_detector.get_layer_dims(self.detection_fv_layer)

        elif self.detection_model=='FasterRCNN':
            self.model_detector = FasterRCNN([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu
            self._w, self._h, self._c = 1, 1, 512


    def load_tracker_model(self):
        raise NotImplementedError


    def load_data_generators(self):
        raise NotImplementedError


    def train(self):
        train_batch, valid_batch = self.load_data_generators()

        model_checkpointer = ModelCheckpoint(
                                        filepath=self.saved_model_path + '-CHKPNT-{epoch:02d}-{val_loss:.2f}.hdf5',
                                        monitor='val_loss',
                                        mode='min',
                                        save_weights_only=False,
                                        save_best_only=False,
                                        verbose=2)

        earlystopper = EarlyStopping(
                                monitor='val_loss',
                                mode='min',
                                patience=10,
                                verbose=2)

        reduce_lr_loss = ReduceLROnPlateau(
                                    monitor='val_loss',
                                    factor=0.5,
                                    patience=5,
                                    verbose=2,
                                    epsilon=1e-4,
                                    mode='min')

        tensorboard = TensorBoard(
                            log_dir=self.tensorboard_dir,
                            histogram_freq=0,
                            batch_size=self.batch_size,
                            write_graph=True,
                            write_grads=False,
                            write_images=False)

        self.model_tracker.fit_generator(
                    generator        = train_batch,
                    steps_per_epoch  = len(train_batch),
                    epochs           = self.max_epochs,
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [earlystopper, model_checkpointer, tensorboard, reduce_lr_loss],
                    max_queue_size   = 3)
