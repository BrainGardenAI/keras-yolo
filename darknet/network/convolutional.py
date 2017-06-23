from keras.layers import Layer, Conv2D, Activation
import keras.backend as K
from keras.engine import InputSpec
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

from cfg_reader import get_activation


class Convolutional(Layer):
    """
    Darknet "convolutional" layer.
    
    Differs from keras' "Conv2D" layer - has optional batch normalization inside.
    """
    def __init__(self, filters=1, size=1, stride=1, 
            batch_normalize=0, pad=0, activation="linear", **params):
        super(Convolutional, self).__init__(**params)
        data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format must be in '
                             '{"channels_last", "channels_first"}')
        self.data_format = data_format
        axis = -1 if self.data_format == 'channels_last' else 1
        
        self.batch_normalize = batch_normalize
        self.padding = "same" if pad else "valid"
        self.filters = filters
        self.kernel_size = size if isinstance(size, tuple) else (size, size)
        self.strides = stride
        self.pad = pad
        #return Conv2D(
        #    #filters=params.get('filters', 1),
        #    #kernel_size=params.get('size', 1),
        #    #strides=params.get('stride', 1),
        #    #padding=padding,
        #    activation=activation) 
        
        if self.batch_normalize: #don't use biases in conv layer, use them in batchnorm
            self.convolutional_layer = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding, 
                use_bias=False)
            self.batchnorm_layer = BatchNormalization(
                axis=axis, 
                center=True, 
                scale=True, 
                epsilon=0.000001)
        else: #use biases in conv layer
            self.convolutional_layer = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding)
        self.activation_layer = get_activation(params.get('activation', 'linear'))
        
    
    def build(self, input_shape):
        super(Convolutional, self).build(input_shape) 
        self.convolutional_layer.build(input_shape)
        output_shape = self.convolutional_layer.compute_output_shape(input_shape)
        
        if self.batch_normalize:
            self.batchnorm_layer.build(output_shape)
            
        self.activation_layer.build(output_shape)
        
    
    def call(self, x, training=None):
        output = self.convolutional_layer.call(x)
        
        if self.batch_normalize:
            output = self.batchnorm_layer.call(output)
            
        output = self.activation_layer.call(output)
        return output
    
    def compute_output_shape(self, input_shape):
        shape = self.convolutional_layer.compute_output_shape(input_shape)
        return shape
        #return self.batchnorm_layer.compute_output_shape(shape) 
        # suppose that the shape doesn't change during activation or batchnorm
        
    def set_weights(self, weights_data):
        if self.batch_normalize:
            (weights, scales, biases, rolling_mean, rolling_variance) = weights_data
            self.convolutional_layer.set_weights((weights,))
            self.batchnorm_layer.set_weights((scales, biases, rolling_mean, rolling_variance))
        else:
            self.convolutional_layer.set_weights(weights_data)
        
    def get_weights(self):
        if self.batch_normalize:
            return self.convolutional_layer.get_weights() + self.batchnorm_layer.get_weights()
        return self.convolutional_layer.get_weights()