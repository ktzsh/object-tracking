Modify Darknet Code
network.c
float *get_output_layer(int n, network *net)
{
    layer l = net.layers[net.n-1];
    return l.output;
}

network.h
float *get_output_layer(int n, network *net);

darknet.py

layer_output = lib.get_output_layer
layer_output.argtypes = [c_int, c_void_p]
layer_output.restype = POINTER(c_float)

layer_output(5, net)
