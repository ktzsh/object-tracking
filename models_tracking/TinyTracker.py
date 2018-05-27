import numpy as np
import tensorflow as tf
from BaseTracker import BaseTracker
from utility.preprocessing import parse_annotation, BatchSequenceGenerator2

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input, LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape, GlobalMaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
#  --------------------------------------------------------------------------------------------------------------------

class TinyTracker(BaseTracker):
    def __init__(self):
        super(TinyTracker, self).__init__()

        self.LSTM_UNITS         = self.config["model_tracker"]["lstm_units"]
        self.SEQUENCE_LENGTH    = self.config["model_tracker"]["sequence_length"]
        self.load_tracker_model()


    def load_tracker_model(self):
        img_input = Input(shape=(self.SEQUENCE_LENGTH, self._w, self._h, self._c), dtype='float32', name='image_fv_input')
        det_input = Input(shape=(self.SEQUENCE_LENGTH, 4), dtype='float32', name='detection_bbox_input')

        if self.pool=='Max':
            x = TimeDistributed(MaxPooling2D((4, 4), strides=(4, 4), name='pool1'))(img_input)
            x = TimeDistributed(Flatten(name='flatten'))(x)
        elif self.pool=='Global':
            x = TimeDistributed(GlobalMaxPooling2D(name='pool1'))(img_input)
        x = concatenate([x, det_input])

        x = LSTM(self.LSTM_UNITS, return_sequences=True, implementation=2, name='recurrent_layer')(x)
        output = TimeDistributed(Dense(4, activation='sigmoid', name='output'))(x)

        self.model_tracker = Model(inputs=[img_input, det_input], outputs=output)
        self.model_tracker.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, decay=0.0), metrics=['loss'])
        self.model_tracker.summary()

    def load_data_generators(self):
        generator_config = {
            'IMAGE_FV_H'            : self._h,
            'IMAGE_FV_W'            : self._w,
            'IMAGE_FV_C'            : self._c,
            'OUTPUT_SHAPE'          : (4,),
            'HEATMAP_SIZE'          : None,
            'DETECTION_FV_LAYER'    : self.detection_fv_layer,
            'DETECTOR'              : self.model_detector,
            'BATCH_SIZE'            : self.batch_size,
            'SEQUENCE_LENGTH'       : self.sequence_length
        }

        train_imgs, seen_train_labels = parse_annotation(self.train_annot_folder, self.train_image_folder, labels=self.classes)
        valid_imgs, seen_valid_labels = parse_annotation(self.val_annot_folder, self.val_image_folder, labels=self.classes)

        train_batch = BatchSequenceGenerator2(train_imgs, generator_config, shuffle=True, augment=False)
        valid_batch = BatchSequenceGenerator2(valid_imgs, generator_config, shuffle=False, augment=False)

        return train_batch, valid_batch
