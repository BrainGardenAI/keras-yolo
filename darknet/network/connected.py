from keras.layers import Dense, Layer
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
    def __init__(self, units, activation=None, **kwargs):
        super(Connected, self).__init__(**kwargs)
        self.dense_layer = Dense(units, activation=activation, **kwargs)
    
    
    def build(self, input_shape):
        super(Connected, self).build(input_shape) 
        densed_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.dense_layer.build(densed_shape)
        
    
    def call(self, x, training=None):
        flatten_inputs = K.batch_flatten(x)
        return self.dense_layer.call(flatten_inputs)
        
        
    def compute_output_shape(self, input_shape):
        dense_input_shape = (input_shape[0], np.prod(input_shape[1:]))
        return self.dense_layer.compute_output_shape(dense_input_shape)
    
    def set_weights(self, weights):
        self.dense_layer.set_weights(weights)
    
    def get_weights(self):
        return self.dense_layer.get_weights()

        