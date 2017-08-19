import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input, Embedding, LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
import h5py


#  --------------------------------------------------------------------------------------------------------------------


def get_model():

    lstm_units = 512
    sequence_length = 4
    heatmap_size = 64*64
    vis_feat_size = 1024

    x_input_vis = Input(shape=(sequence_length, 19, 19, vis_feat_size), dtype='float32', name='img_vis_map')
    x_vis = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(x_input_vis)
    x_vis = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'))(x_vis)
    x_vis = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x_vis)
    x_vis = TimeDistributed(Flatten(name='flatten'))(x_vis)
    x_vis_feat = TimeDistributed(Dense(4096, activation='relu', name='vis_feat'))(x_vis)
    feature_extractor = Model(inputs=x_input_vis, outputs=x_vis_feat)


    x_input_feat_history = Input(shape=(sequence_length, 19, 19, vis_feat_size), dtype='float32', name='img_vis_map')
    x_input_feat_next = Input(shape=(1, 19, 19, vis_feat_size), dtype='float32', name='input_feat_layer_next')
    x_input_heat_feat = Input(shape=(sequence_length, heatmap_size), dtype='float32', name='img_heatmap_feat')

    x_vis_feat = feature_extractor(x_input_feat_history)
    x_vis_feat = LSTM(lstm_units, return_sequences=False, implementation=2, name='recurrent_layer')(x_vis_feat)

    x_vis_feat_next = feature_extractor(x_input_feat_next)
    x_vis_feat_next = Reshape((-1))(x_vis_feat_next)

    x = concatenate([x_vis_feat, x_vis_feat_next], name='concat')
    output = Dense(heatmap_size, activation='sigmoid', name='location')(x)

    model = Model(inputs=[x_input_feat_history, x_input_feat_next, x_input_heat_feat], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train(x_train_vis, x_train_vis_next, x_train_heat, y_train, x_val_vis, x_val_vis_next, x_val_heat, y_val):

    model = get_model()
    # model = load_model('./weights/TB-50_WEIGHTS/weights.13-0.03.hdf5')
    checkpointer = ModelCheckpoint(filepath='./weights/TB-50_WEIGHTS/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
    model.fit([x_train_vis, x_train_vis_next, x_train_heat], y_train, validation_data=([x_val_vis, x_val_vis_next, x_val_heat],y_val), batch_size=32, epochs=200, verbose=1, shuffle=False, callbacks=[checkpointer, earlystopper])


def test(x_test, x_test_next, x_heat, y_test):
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=False))
    with sess.as_default():
        model = get_model()
        model.load_weights('weights/TB-50_WEIGHTS/weights.25-0.02.hdf5')
        y_out = model.predict([x_test, x_test_next, x_heat], batch_size=32)
        return y_out





def get_model_simple():

    lstm_units = 512
    sequence_length = 4
    heatmap_size = 64*64
    vis_feat_size = 1024

    x_input = Input(shape=(sequence_length, heatmap_size + vis_feat_size), dtype='float32', name='input_feat_layer')
    x_input_next = Input(shape=(vis_feat_size), dtype='float32', name='input_feat_layer_next')
    x_history = LSTM(lstm_units, return_sequences=False, implementation=2, name='recurrent_layer')(x_input)
    x = concatenate([x_history, x_input_next], name='concat')
    output = Dense(heatmap_size, activation='sigmoid', name='location')(x)

    model = Model(inputs=[x_input, x_input_next], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_simple(x_train, x_train_next, y_train, x_val, x_val_next, y_val):

    model = get_model_simple()
    checkpointer = ModelCheckpoint(filepath='weights/TB-50_WEIGHTS/weights_simple.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
    model.fit([x_train, x_train_next], y_train, validation_data=([x_val, x_val_next], y_val), batch_size=64, epochs=200, verbose=1, shuffle=False, callbacks=[checkpointer, earlystopper])

def test_simple(x_test, x_test_next, y_test):
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=False))
    with sess.as_default():
        model = get_model_simple()
        model.load_weights('weights/TB-50_WEIGHTS/weights_simple.40-0.02.hdf5')
        y_out = model.predict([x_test, x_test_next], batch_size=32)
        return y_out
