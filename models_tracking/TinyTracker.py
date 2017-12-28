import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input, LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences


#  --------------------------------------------------------------------------------------------------------------------

class TinyTracker:

    _w         = None
    _h         = None
    _c         = None
    tgpu_id    = 1
    dgpu_id    = 2
    cpu_mode   = 0
    pool       = 'none'
    batch_size = 32
    max_epochs = 40

    DETECTION_FV_LAYER = 25
    INPUT_FEAT         = 1024
    LSTM_UNITS         = 1024
    SEQUENCE_LENGTH    = 6
    model_tracker      = None
    model_detector     = None
    detection_model    = None

    def __init__(self, argvs):
        self.argv_parser(argvs)
        self.load_detection_model()
        self.load_tracker_model()


    def argv_parser(self, argvs):
        self.detection_model = argv[0]
        self.cpu_mode = argv[1]
        self.tgpu_id = argv[2]
        self.dgpu_id = argv[3]
        self.pool = argv[4]
        self.batch_size = argv[5]
        self.max_epochs = argv[6]
        if self.tgpu_id==self.dgpu_id:
            print '#####################################################################################'
            print '# [WARNING] Models for Detection and Tracking should be loaded on DIFFERENT GPUs OR #'
            print '# save the extracted features from detection network first then run the tracker.    #'
            print '#####################################################################################'


    def load_detection_model(self):
        if self.detection_model=='YOLO':
            self.model_detector = YOLO([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu
            self.INPUT_FEAT = 2048
            self.LSTM_UNITS = 512
            self.SEQUENCE_LENGTH = 6
            self.DETECTION_FV_LAYER = 29
            self._w = 19
            self._h = 19
            self._c = 1024

        elif self.detection_model=='FasterRCNN':
            self.model_detector = FasterRCNN([self.cpu_mode, self.dgpu_id]) #load detection nework on other gpu
            self.INPUT_FEAT = 2048
            self.LSTM_UNITS = 512
            self.SEQUENCE_LENGTH = 6
            self.DETECTION_FV_LAYER = 'relu5_3'
            self._w = 1 #CORRECT THIS
            self._h = 1 #CORRECT THIS
            self._c = 512


    def load_tracker_model(self):
        img_input = Input(shape=(self.SEQUENCE_LENGTH, self._w, self._h, self._c), dtype='float32', name='image_fv_input')
        det_input = Input(shape=(self.SEQUENCE_LENGTH, 4), dtype='float32', name='detection_bbox_input')

        if self.pool=='Max':
            x = TimeDistributed(MaxPooling2D((4, 4), strides=(4, 4), name='pool1'))(img_input)
            x = TimeDistributed(Flatten(name='flatten'))(x)
        elif self.pool=='Global':
            x = TimeDistributed(GlobalMaxPooling2D(name='pool1')(x))
        x = TimeDistributed(Dense(self.INPUT_FEAT, activation='relu', name='fc1'))(x)

        x = concatenate([x, det_input])

        x = LSTM(self.LSTM_UNITS, return_sequences=True, implementation=2, name='recurrent_layer')(x)
        output = TimeDistributed(Dense(4, activation='sigmoid', name='output'))(x)

        model = Model(inputs=x_input, outputs=output)
        optimizer = Adam(lr=0.01, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model_tracker = model
        model.summary()


    def get_batch(self, frame_dim_dirs, frame_paths_dirs, frame_bboxs_dirs):
        idx = 0
        x_img = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, self._w, self._h, self._c), dtype='float32')
        x_bbox = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, 4), dtype='float32')
        y_bbox = np.zeros((self.batch_size, self.SEQUENCE_LENGTH, 4), dtype='float32')

        while 1:
            for i, (frame_paths, frame_bboxs, frame_dim) in enumerate(zip(frame_paths_dirs, frame_bboxs_dirs, frame_dim_dirs)):
                frame_height, frame_width = frame_dim[0], frame_dim[1]
                sequence_size = len(frame_paths)

                for j in range(sequence_size - self.SEQUENCE_LENGTH):
                    frame_path, frame_bbox = frame_paths[j], frame_bboxs[j]

                    obj_detections, vis_feat = self.model_detector.extract_spatio_info(frame_path, self.DETECTION_FV_LAYER)
                    vis_feat = vis_feat.reshape((self._w, self._h, self._c))

                    print "BBOX FROM DETECTION", obj_detections

                    for k in range(self.SEQUENCE_LENGTH):
                        det_x = (frame_bboxs[j+k][0] + frame_bboxs[j+k][2]/2.0) / frame_width
                        det_y = (frame_bboxs[j+k][1] + frame_bboxs[j+k][3]/2.0) / frame_height
                        det_w = (frame_bboxs[j+k][2]) / frame_width
                        det_h = (frame_bboxs[j+k][3]) / frame_height

                        x_img[idx, k, :, :, :] = vis_feat
                        x_dets[idx, k, :] = None #CORRECT THIS
                        y_dets[idx, k, :] = np.array([det_x, det_y, det_w, det_h], dtype='float32')

                    idx = (idx + 1) % self.batch_size
                    if idx==0:
                        yield ([x_img, x_bbox], y_bbox)


    def train(self, train_data_dirs, val_data_dirs):
        t_frame_dim_dirs, t_frame_paths_dirs, t_frame_bboxs_dirs = train_data_dirs
        v_frame_dim_dirs, v_frame_paths_dirs, v_frame_bboxs_dirs = val_data_dirs

        t_data_size = sum( len(t) - self.SEQUENCE_LENGTH + 1 for t in t_frame_paths_dirs)
        v_data_size = sum( len(v) - self.SEQUENCE_LENGTH + 1 for v in v_frame_paths_dirs)

        model_checkpointer = ModelCheckpoint(filepath='weights/MODEL_TinyTracker_' + self.detection_model + '.h5',
                                        monitor='val_loss',
                                        mode='min',
                                        save_weights_only=False,
                                        save_best_only=False,
                                        verbose=0)

        weights_checkpointer = ModelCheckpoint(filepath='weights/WEIGHTS_TinyTracker_' + self.detection_model + '.h5',
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

        csv_logger = CSVLogger('logs/TinyTracker_' + self.detection_model + '.log', append=False)

        self.model_tracker.fit_generator(
                            self.get_batch(t_frame_dim_dirs, t_frame_paths_dirs, t_frame_bboxs_dirs),
                            steps_per_epoch=t_data_size//self.batch_size,
                            epochs=self.max_epochs,
                            callbacks=[earlystopper, model_checkpointer, weights_checkpointer, csv_logger, reduce_lr_loss],
                            validation_data=self.get_batch(v_frame_dim_dirs, v_frame_paths_dirs, v_frame_bboxs_dirs),
                            validation_steps=v_data_size//self.batch_size
                            )
