from keras import backend as K
from keras import activations
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.models import Model
from keras.utils import conv_utils

from recurrentshop.engine import RNNCell

class ExtendedConvLSTM2DCell(RNNCell):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 **kwargs):
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(ExtendedConvLSTM2DCell, self).__init__(**kwargs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(ExtendedConvLSTM2DCell, self).get_config()
        config.update(base_config)
        return config

class ConvLSTM2DCell(ExtendedConvLSTM2DCell):

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def build_model(self, input_shape):

        rows = input_shape[1]
        cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows,
                                             self.kernel_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols,
                                             self.kernel_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate[1])


        output_shape = (input_shape[0], rows, cols, self.filters)
        input_dim = input_shape[-1]

        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)
        kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # bias = None
        if self.use_bias:
            bias = self.add_weight(
                shape=(self.filters * 4,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            if self.unit_forget_bias:
                bias_value = np.zeros((self.filters * 4,))
                bias_value[self.filters: self.filters * 2] = 1.
                K.set_value(bias, bias_value)
        else:
            bias = None


        kernel_i = kernel[:, :, :, :self.filters]
        recurrent_kernel_i = recurrent_kernel[:, :, :, :self.filters]
        kernel_f = kernel[:, :, :, self.filters: self.filters * 2]
        recurrent_kernel_f = recurrent_kernel[:, :, :, self.filters: self.filters * 2]
        kernel_c = kernel[:, :, :, self.filters * 2: self.filters * 3]
        recurrent_kernel_c = recurrent_kernel[:, :, :, self.filters * 2: self.filters * 3]
        kernel_o = kernel[:, :, :, self.filters * 3:]
        recurrent_kernel_o = recurrent_kernel[:, :, :, self.filters * 3:]

        if self.use_bias:
            bias_i = self.bias[:self.filters]
            bias_f = self.bias[self.filters: self.filters * 2]
            bias_c = self.bias[self.filters * 2: self.filters * 3]
            bias_o = self.bias[self.filters * 3:]
        else:
            bias_i = None
            bias_f = None
            bias_c = None
            bias_o = None


        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)

        x_i = self.input_conv(x, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(x, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(x, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(x, kernel_o, bias_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1, recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        return Model([x, h_tm1, c_tm1], [h, h, c])
