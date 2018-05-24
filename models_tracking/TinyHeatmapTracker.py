import numpy as np
import tensorflow as tf
from BaseTracker import BaseTracker
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input, LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape, GlobalMaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
#  --------------------------------------------------------------------------------------------------------------------

class TinyHeatmapTracker(BaseTracker):

    INPUT_FEAT         = 512
    LSTM_UNITS         = 512
    HEATMAP_SIZE       = 32
    SEQUENCE_LENGTH    = 6

    model_tracker      = None
    model_detector     = None
    detection_model    = None

    def __init__(self, argv):
        super(TinyHeatmapTracker, self).__init__(argv)
        self.load_tracker_model()


    def load_tracker_model(self):
        img_input = Input(shape=(self.SEQUENCE_LENGTH, self._w, self._h, self._c), dtype='float32', name='image_fv_input')
        det_input = Input(shape=(self.SEQUENCE_LENGTH, (self.HEATMAP_SIZE**2)), dtype='float32', name='detection_bbox_input')

        if self.pool=='Max':
            x = TimeDistributed(MaxPooling2D((4, 4), strides=(4, 4), name='pool1'))(img_input)
            x = TimeDistributed(Flatten(name='flatten'))(x)
        elif self.pool=='Global':
            x = TimeDistributed(GlobalMaxPooling2D(name='pool1'))(img_input)
        x = TimeDistributed(Dense(self.INPUT_FEAT, activation='relu', name='fc1'))(x)

        x = concatenate([x, det_input])

        x = LSTM(self.LSTM_UNITS, return_sequences=True, implementation=2, name='recurrent_layer')(x)
        output = TimeDistributed(Dense((self.HEATMAP_SIZE**2), activation='sigmoid', name='output'))(x)

        model = Model(inputs=[img_input, det_input], outputs=output)
        optimizer = Adam(lr=0.01, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model_tracker = model
        model.summary()


    def get_batch(self, argv):
        idx = 0
        frame_dim_dirs = argv[0]
        frame_paths_dirs = argv[1]
        frame_bboxs_dirs = argv[2]

        x_img = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, self._w, self._h, self._c), dtype='float32')
        x_bbox = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, (self.HEATMAP_SIZE**2)), dtype='float32')
        y_bbox = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, (self.HEATMAP_SIZE**2)), dtype='float32')

        while 1:
            for i, (frame_paths, frame_bboxs, frame_dim) in enumerate(zip(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs)):
                frame_height, frame_width = frame_dim[0], frame_dim[1]
                sequence_size = len(frame_paths)

                for j in range(sequence_size - self.SEQUENCE_LENGTH):
                    frame_path, frame_bbox = frame_paths[j], frame_bboxs[j]

                    obj_detections, vis_feat = self.model_detector.extract_spatio_info(frame_path, self.detection_fv_layer)
                    vis_feat = vis_feat.reshape((self._w, self._h, self._c))

                    print "BBOX FROM DETECTION", obj_detections

                    for k in range(self.SEQUENCE_LENGTH):
                        obj_det, vis_feat = self.model_detector.extract_spatio_info(frame_paths[j+k], self.detection_fv_layer)
                        vis_feat = vis_feat.reshape((self._w, self._h, self._c))

                        det_x = (frame_bboxs[j+k][0] + frame_bboxs[j+k][2]/2.0) / frame_width
                        det_y = (frame_bboxs[j+k][1] + frame_bboxs[j+k][3]/2.0) / frame_height
                        det_w = (frame_bboxs[j+k][2]) / frame_width
                        det_h = (frame_bboxs[j+k][3]) / frame_height

                        det_x_in = (obj_det[0][2][0]) / frame_width
                        det_y_in = (obj_det[0][2][1]) / frame_height
                        det_w_in = (obj_det[0][2][2]) / frame_width
                        det_h_in = (obj_det[0][2][3]) / frame_height

                        x_img[idx, k, :, :, :] = vis_feat
                        x_bbox[idx, k, :] = generate_heatmap_feat(det_x_in - det_w_in/2.0, det_y_in - det_h_in/2.0, det_w_in, det_h_in, hmap_size=self.HEATMAP_SIZE)
                        y_bbox[idx, k, :] = generate_heatmap_feat(det_x - det_w/2.0, det_y - det_h/2.0, det_w, det_h, hmap_size=self.HEATMAP_SIZE)

                    idx = (idx + 1) % self.batch_size
                    if idx==0:
                        yield ([x_img, x_bbox], y_bbox)
