from .ConvLSTM2D import ConvLSTM2DCell

from keras.layers import *
from keras.models import Model

input = Input((5,))
state1_tm1 = Input((10,10,32))
state2_tm1 = Input((10,10,32))
state3_tm1 = Input((10,10,32))

lstm_output, state1_t, state2_t = LSTMCell(10)([input, state1_tm1, state2_tm1])
gru_output, state3_t = GRUCell(10)([input, state3_tm1])

output = add([lstm_output, gru_output])
output = Activation('tanh')(output)

rnn = RecurrentModel(input=input, initial_states=[state1_tm1, state2_tm1, state3_tm1], output=output, final_states=[state1_t, state2_t, state3_t])
