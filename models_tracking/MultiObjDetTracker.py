import pickle
import os, cv2
import imgaug as ia
from tqdm import tqdm
import numpy as np
from imgaug import augmenters as iaa
from utils.preprocessing import parse_annotation, create_sequences_from_parsed_annotations, BatchGenerator, BatchSequenceGenerator
from utils.utils import WeightReader, decode_netout, draw_boxes, normalize

import tensorflow as tf
import keras.backend as K
K.set_learning_phase(1)

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, ConvLSTM2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate

import matplotlib.pyplot as plt


IMAGENET_LABEL_MAP = {
                        'n02691156' : 'airplane',
                        'n02419796' : 'antelope',
                        'n02131653' : 'bear',
                        'n02834778' : 'bicycle',
                        'n01503061' : 'bird',
                        'n02924116' : 'bus',
                        'n02958343' : 'car',
                        'n02402425' : 'cattle',
                        'n02084071' : 'dog',
                        'n02121808' : 'domestic cat',
                        'n02503517' : 'elephant',
                        'n02118333' : 'fox',
                        'n02510455' : 'giant panda',
                        'n02342885' : 'hamster',
                        'n02374451' : 'horse',
                        'n02129165' : 'lion',
                        'n01674464' : 'lizard',
                        'n02484322' : 'monkey',
                        'n03790512' : 'motorcycle',
                        'n02324045' : 'rabbit',
                        'n02509815' : 'red panda',
                        'n02411705' : 'sheep',
                        'n01726692' : 'snake',
                        'n02355227' : 'squirrel',
                        'n02129604' : 'tiger',
                        'n04468005' : 'train',
                        'n01662784' : 'turtle',
                        'n04530566' : 'watercraft',
                        'n02062744' : 'whale',
                        'n02391049' : 'zebra'
                    }

class MultiObjDetTracker:

    TRAIN_MODE = 'Tracker' #or Detector

    LABELS_COCO = [
                'person',        'bicycle',    'car',           'motorcycle',    'airplane',     'bus',
                'train',         'truck',      'boat',          'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench',      'bird',          'cat',           'dog',          'horse',
                'sheep',         'cow',        'elephant',      'bear',          'zebra',        'giraffe',
                'backpack',      'umbrella',   'handbag',       'tie',           'suitcase',     'frisbee',
                'skis',          'snowboard',  'sports ball',   'kite',          'baseball bat', 'baseball glove',
                'skateboard',    'surfboard',  'tennis racket', 'bottle',        'wine glass',   'cup',
                'fork',          'knife',      'spoon',         'bowl',          'banana',       'apple',
                'sandwich',      'orange',     'broccoli',      'carrot',        'hot dog',      'pizza',
                'donut',         'cake',       'chair',         'couch',         'potted plant', 'bed',
                'dining table',  'toilet',     'tv',            'laptop',        'mouse',        'remote',
                'keyboard',      'cell phone', 'microwave',     'oven',          'toaster',      'sink',
                'refrigerator',  'book',       'clock',         'vase',          'scissors',     'teddy bear',
                'hair drier',    'toothbrush'
            ]


    LABELS_IMAGENET_VIDEO = [
                                'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
                                'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
                                'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
                                'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
                                'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049'
                            ]

    LABELS           = LABELS_COCO
    IMAGE_H, IMAGE_W = 416, 416
    GRID_H,  GRID_W  = 13 , 13
    BOX              = 5
    CLASS            = len(LABELS)
    CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
    # OBJ_THRESHOLD    = 0.3#0.5
    # NMS_THRESHOLD    = 0.3#0.45
    OBJ_THRESHOLD    = 0.5
    NMS_THRESHOLD    = 0.45
    ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    NO_OBJECT_SCALE  = 1.0
    OBJECT_SCALE     = 5.0
    COORD_SCALE      = 1.0
    CLASS_SCALE      = 1.0

    BATCH_SIZE       = 1
    WARM_UP_BATCHES  = 0
    TRUE_BOX_BUFFER  = 50

    SEQUENCE_LENGTH   = 4
    MAX_BOX_PER_IMAGE = 50


    weight_path = 'darknet/yolo.weights'
    train_image_folder = None
    train_annot_folder = None
    valid_image_folder = None
    valid_annot_folder = None

    model = None
    model_detector = None
    model_tracker = None

    def __init__(self, argv):
        self.TRAIN_MODE = argv[0]
        if self.TRAIN_MODE=='Tracker':
            self.LABELS             = self.LABELS_IMAGENET_VIDEO
            self.CLASS              = len(self.LABELS)
            self.CLASS_WEIGHTS      = np.ones(self.CLASS, dtype='float32')
            self.train_image_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Data/VID/train/'
            self.train_annot_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Annotations/VID/train/'
            self.valid_image_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Data/VID/val/'
            self.valid_annot_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Annotations/VID/val/'
            self.load_detector_model()
            self.load_tracker_model()
            self.model = self.model_tracker

        elif self.TRAIN_MODE=='Detector':
            self.LABELS             = self.LABELS_COCO
            self.CLASS              = len(self.LABELS)
            self.CLASS_WEIGHTS      = np.ones(self.CLASS, dtype='float32')
            self.train_image_folder = 'data/coco/train2014/'
            self.train_annot_folder = 'data/coco/train2014ann/'
            self.valid_image_folder = 'data/coco/val2014/'
            self.valid_annot_folder = 'data/coco/val2014ann/'
            self.load_detector_model()
            self.model = self.model_detector


    def loss_fxn(self, y_true, y_pred, tboxes, message=''):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.GRID_W), [self.GRID_H]), (1, self.GRID_H, self.GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.BATCH_SIZE, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.ANCHORS, [1,1,1,self.BOX,2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.COORD_SCALE

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6

        # tboxes_wh = None
        # tboxes_xy = None
        # if self.TRAIN_MODE=='Tracker':
        #     tboxes_xy = self.true_boxes[timestep-1, ..., 0:2]
        #     tboxes_wh = self.true_boxes[timestep-1, ..., 2:4]
        # elif self.TRAIN_MODE=='Detector':
        #     tboxes_xy = self.true_boxes[..., 0:2]
        #     tboxes_wh = self.true_boxes[..., 2:4]
        #
        # true_xy = tboxes_xy
        # true_wh = tboxes_wh
        true_xy = tboxes[..., 0:2]
        true_wh = tboxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.OBJECT_SCALE

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.CLASS_WEIGHTS, true_box_class) * self.CLASS_SCALE

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.COORD_SCALE/2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.WARM_UP_BATCHES),
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(self.ANCHORS, [1,1,1,self.BOX,2]) * no_boxes_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy,
                                       true_box_wh,
                                       coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

        """
        Debugging code
        """
        current_recall = nb_pred_box/(nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall)

        # loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [tf.zeros((1))], message=('LOSS INFO'), summarize=1000)
        loss = tf.Print(loss, [loss_xy], message=message+'Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message=message+'Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [loss_conf], message=message+'Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [loss_class], message=message+'Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message=message+'Total Loss \t', summarize=1000)
        loss = tf.Print(loss, [current_recall], message=message+'Current Recall', summarize=1000)
        loss = tf.Print(loss, [total_recall/seen], message=message+'Average Recall', summarize=1000)
        return loss

    def custom_loss_detect(self, y_true, y_pred):
        return self.loss_fxn(y_true, y_pred, tboxes=self.true_boxes)


    def custom_loss_dtrack(self, y_true, y_pred):
        tboxes    = self.true_boxes
        new_shape = self.BATCH_SIZE * self.SEQUENCE_LENGTH

        y_pred = tf.reshape(y_pred, (new_shape, self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))
        y_true = tf.reshape(y_true, (new_shape, self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))
        tboxes = tf.reshape(tboxes, (new_shape, 1, 1, 1, self.TRUE_BOX_BUFFER , 4))

        loss = self.loss_fxn(y_true, y_pred, tboxes=tboxes, message='[DETECTOR]')
        return loss

    def custom_loss_ttrack(self, y_true, y_pred):
        tboxes    = self.true_boxes
        new_shape = self.BATCH_SIZE * self.SEQUENCE_LENGTH

        y_pred = tf.reshape(y_pred, (new_shape, self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))
        y_true = tf.reshape(y_true, (new_shape, self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))
        tboxes = tf.reshape(tboxes, (new_shape, 1, 1, 1, self.TRUE_BOX_BUFFER , 4))

        loss = self.loss_fxn(y_true, y_pred, tboxes=tboxes, message='[TRACKER]')
        return loss

    def init_weights(self):
        weight_reader = WeightReader(self.weight_path)
        weight_reader.reset()
        nb_conv = 23

        for i in range(1, nb_conv+1):
            conv_layer = self.model_detector.get_layer('conv_' + str(i))

            if i < nb_conv:
                norm_layer = self.model_detector.get_layer('norm_' + str(i))

                size = np.prod(norm_layer.get_weights()[0].shape)

                beta  = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean  = weight_reader.read_bytes(size)
                var   = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights()) > 1:
                bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel])


    def load_detector_model(self):

        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        input_image = Input(batch_shape=(self.BATCH_SIZE, self.IMAGE_H, self.IMAGE_W, 3))
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x_vis = LeakyReLU(alpha=0.1)(x)

        # Layer 23
        x_bbox = Conv2D(self.BOX * (4 + 1 + self.CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x_vis)

        if self.TRAIN_MODE=='Detector':
            x_bbox = Reshape((self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))(x_bbox)
            true_boxes  = Input(shape=(1, 1, 1, self.TRUE_BOX_BUFFER , 4))
            output_det = Lambda(lambda args: args[0])([x_bbox, true_boxes])
            self.model_detector = Model([input_image, true_boxes], output_det)

        elif self.TRAIN_MODE=='Tracker':
            self.model_detector = Model(input_image, [x_bbox, x_vis])

        self.model_detector.summary()
        self.init_weights()


    def load_tracker_model(self):

        input_images = Input(batch_shape=(self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.IMAGE_H, self.IMAGE_W, 3), name='images_input')

        outputs, names = [], ['timedist_bbox', 'timedist_vis']
        for i, out in enumerate(self.model_detector.output):
            outputs.append(TimeDistributed(Model(self.model_detector.input, out), name=names[i])(input_images))
        x_bbox, x_vis = outputs

        output_det = TimeDistributed(Reshape((self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS)), name='detection')(x_bbox)

        z = concatenate([x_bbox, x_vis])
        z_vis = ConvLSTM2D(1024, (3,3), strides=(1,1), padding='same', return_sequences=True, name='tconv_lstm')(z)

        # z = TimeDistributed(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False, name='tconv_1'), name='timedist_tconv1')(z)
        # z = TimeDistributed(BatchNormalization(name='tnorm_1'), name='timedist_tnorm')(z)
        # z_vis = TimeDistributed(LeakyReLU(alpha=0.1))(z)

        z_bbox = TimeDistributed(Conv2D(self.BOX * (4 + 1 + self.CLASS), (1,1), strides=(1,1), padding='same', name='tconv_2'), name='timedist_tconv2')(z_vis)
        z_out = TimeDistributed(Reshape((self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS)))(z_bbox)

        self.true_boxes = Input(batch_shape=(self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1, 1, 1, self.TRUE_BOX_BUFFER , 4), name='bbox_input')
        output_trk = Lambda(lambda args: args[0], name='tracking')([z_out, self.true_boxes])

        self.model_tracker = Model([input_images, self.true_boxes], [output_trk, output_det], name='tracker')
        self.model_tracker.summary()


    def load_data_generators(self, generator_config):
        train_imgs   = None
        valid_imgs   = None
        train_batch  = None
        valid_batch  = None

        pickle_train = 'data/MultiObjDetTracker_TrainAnn' + self.TRAIN_MODE + '.pickle'
        pickle_val   = 'data/MultiObjDetTracker_ValAnn' + self.TRAIN_MODE + '.pickle'

        if os.path.isfile(pickle_train):
            with open (pickle_train, 'rb') as fp:
               train_imgs = pickle.load(fp)
        else:
            train_imgs, seen_train_labels = parse_annotation(self.train_annot_folder, self.train_image_folder, labels=self.LABELS)
            with open(pickle_train, 'wb') as fp:
               pickle.dump(train_imgs, fp)


        if os.path.isfile(pickle_val):
            with open (pickle_val, 'rb') as fp:
               valid_imgs = pickle.load(fp)
        else:
            valid_imgs, seen_valid_labels = parse_annotation(self.valid_annot_folder, self.valid_image_folder, labels=self.LABELS)
            with open(pickle_val, 'wb') as fp:
               pickle.dump(valid_imgs, fp)


        if self.TRAIN_MODE=='Tracker':
            print "TRAIN GEN", len(train_imgs), generator_config
            train_batch = BatchSequenceGenerator(train_imgs, generator_config, norm=normalize)
            print "VALID GEN", len(valid_imgs), generator_config
            valid_batch = BatchSequenceGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

        elif self.TRAIN_MODE=='Detector':
            train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)
            valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

        return train_batch, valid_batch

    def train(self):

        if  self.TRAIN_MODE=='Detector':  last = -4
        elif self.TRAIN_MODE=='Tracker':  last = -1

        layer   = self.model_detector.layers[last] # the last convolutional layer
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(self.GRID_H * self.GRID_W)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.GRID_H * self.GRID_W)

        layer.set_weights([new_kernel, new_bias])

        generator_config = {
            'IMAGE_H'         : self.IMAGE_H,
            'IMAGE_W'         : self.IMAGE_W,
            'GRID_H'          : self.GRID_H,
            'GRID_W'          : self.GRID_W,
            'BOX'             : self.BOX,
            'LABELS'          : self.LABELS,
            'CLASS'           : len(self.LABELS),
            'ANCHORS'         : self.ANCHORS,
            'BATCH_SIZE'      : self.BATCH_SIZE,
            'TRUE_BOX_BUFFER' : 50,
            'SEQUENCE_LENGTH' : self.SEQUENCE_LENGTH
        }

        train_batch, valid_batch = self.load_data_generators(generator_config)
        print "Length of Generators", len(train_batch), len(valid_batch)

        early_stop = EarlyStopping(monitor   = 'val_loss',
                                   min_delta = 0.001,
                                   patience  = 5,
                                   mode      = 'min',
                                   verbose   = 1)

        checkpoint = ModelCheckpoint('weights/WEIGHTS_MultiObjDetTracker.h5',
                                     monitor        = 'val_loss',
                                     verbose        = 1,
                                     save_best_only = True,
                                     # save_weights_only = True,
                                     mode           = 'min',
                                     period         = 1)

        tb_counter  = len([log for log in os.listdir(os.path.expanduser('./logs/')) if 'MultiObjDetTracker_' in log]) + 1
        tensorboard = TensorBoard(log_dir        = os.path.expanduser('./logs/') + 'MultiObjDetTracker_' + str(tb_counter),
                                  histogram_freq = 0,
                                  write_graph    = True,
                                  write_images   = False)

        optimizer = Adam(lr=0.1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
        #optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

        if self.TRAIN_MODE=='Tracker':
            self.model.compile(loss=[self.custom_loss_ttrack, self.custom_loss_dtrack], loss_weights=[0.5, 0.5], optimizer=optimizer)

            self.model.fit_generator(
                        generator        = train_batch,
                        steps_per_epoch  = len(train_batch),
                        epochs           = 100,
                        verbose          = 1,
                        validation_data  = valid_batch,
                        validation_steps = len(valid_batch),
                        callbacks        = [early_stop, checkpoint, tensorboard],
                        max_queue_size   = 3)

        elif self.TRAIN_MODE=='Detector':
            self.model.compile(loss=self.custom_loss_detect, optimizer=optimizer)
            self.model.fit_generator(
                        generator        = train_batch,
                        steps_per_epoch  = len(train_batch),
                        epochs           = 100,
                        verbose          = 1,
                        validation_data  = valid_batch,
                        validation_steps = len(valid_batch),
                        callbacks        = [early_stop, checkpoint, tensorboard],
                        max_queue_size   = 3)

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def predict(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
        resized_image = normalize(resized_image)
        input_image = resized_image.reshape((1, self.IMAGE_H, self.IMAGE_W, 3))

        dummy_array = np.zeros((1,1,1,1,self.MAX_BOX_PER_IMAGE,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.OBJ_THRESHOLD, self.NMS_THRESHOLD, self.ANCHORS, len(self.LABELS))
        image = draw_boxes(image, boxes, self.LABELS)

        print len(boxes), 'Bounding Boxes Found'
        print "File Saved to Output.jpg"
        cv2.imwrite('Output.jpg', image)
