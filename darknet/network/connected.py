from keras.layers import Layer, Dense
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.engine import InputSpec
import numpy as np


class Connected(Layer):
    """
    Darknet "connected" layer. Main difference vs. keras Dense layer is that 
    input also becomes flatten.
    The same as 
    
    ```
    def get_connected(params):
        activation = get_activation(params.get('activation', "linear"))
        def _connected(x):
            y = Flatten()(x)
            return Dense(params.get('output', 1), activation=activation)(y)
        
        return Lambda(_connected)
    ```
    
    - but also has weights.
    """
    def __init__(self, output=1, activation=None, batch_normalize=0, **kwargs):
        self.units = output
        self.activation = activation
        self.batch_normalize = batch_normalize
        super(Connected, self).__init__(**kwargs)
        self.dense_layer = Dense(self.units, **kwargs)
        # TODO: axis check
        if self.batch_normalize:
            self.batchnorm_layer = BatchNormalization(scale=True, center=False)
    
    
    def build(self, input_shape):
        super(Connected, self).build(input_shape) 
        densed_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.dense_layer.build(densed_shape)
        if self.batch_normalize:
            self.batchnorm_layer.build(self.dense_layer.output_shape(input_shape))
        
    
    def call(self, x, training=None):
        flatten_inputs = K.batch_flatten(x)
        output = self.dense_layer.call(flatten_inputs)
        if self.batch_normalize:
            output = self.batchnorm_layer.call(output)
        return output
        
        
    def compute_output_shape(self, input_shape):
        dense_input_shape = (input_shape[0], np.prod(input_shape[1:]))
        shape = self.dense_layer.compute_output_shape(dense_input_shape)
        if self.batch_normalize:
            shape = self.batch_normalize.compute_output_shape(shape)
        return shape
    
    def set_weights(self, weights):
        if self.batch_normalize:
            (weights, bias, scales, rolling_mean, rolling_variance) = weights
            self.dense_layer.set_weights((weights, bias))
            self.batchnorm_layer.set_weights((scales, rolling_mean, rolling_variance))
        else:
            self.dense_layer.set_weights(weights)
            
    
    def get_weights(self):
        if self.batch_normalize:
            return self.dense_layer.get_weights() + self.batchnorm_layer.get_weights()
        return self.dense_layer.get_weights()

        