"""
There are two versions of detection layer in darknet, one of them is named `region`.
This module contains this layer implemented.
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
#from builtins import range
from keras.activations import softmax


class Region(Layer):
    def __init__(self, coords=4, classes=20, num=1,
            log=0, sqrt=0, softmax=0, background=0, max=30,
            jitter=0.2, 
            rescore = 0, thresh=0.5, classfix=0, absolute=0, random=0,
            coord_scale=1, object_scale=1,
            noobject_scale=1, class_scale=1,
            bias_match=0,
            tree=None,#tree_file for softmax_tree - not used now
            map_filename=None, # file name for map_file - not used
            anchors=None,
            **kwargs
            ):
        super(Region, self).__init__(**kwargs)
        self.coords = coords
        self.classes = classes
        self.num = num
        self.background = background
        print(coords, classes)
        self.c = (self.coords+self.classes+1)*num
        if anchors:
            self.biases = list(map(float, anchors))
        pass
    
    
    def build(self, input_shape):
        self.inputs = np.prod(input_shape[1:])
        self.outputs = self.inputs
        #assert(self.c == input_shape[-1])
        pass
    
    
    def _entry_index(self, n, entry):
        #simplified version - gets last coordinate in input tensor
        return n*(self.coords + self.classes + 1) + entry
        
        
    def _process_input(self, x):
        """Apply logistic and softmax activations to input tensor
        """
        logistic_activate = lambda x: 1.0/(1.0 + K.exp(-x))
        
        (batch, w, h, channels) = x.get_shape()
        x_temp = K.permute_dimensions(x, (3, 0, 1, 2))
        x_t = []
        for i in range(self.num):
            k = self._entry_index(i, 0)
            x_t.extend([
                logistic_activate(K.gather(x_temp, (k, k + 1))), # 0
                K.gather(x_temp, (k + 2, k + 3))])
            if self.background:
                x_t.append(K.gather(x_temp, (k + 4,)))
            else:
                x_t.append(logistic_activate(K.gather(x_temp, (k + 4,))))
                
            x_t.append(
                softmax(
                    K.gather(x_temp, tuple(range(k + 5, k + self.coords + self.classes + 1))),
                    axis=0))
        x_t = K.concatenate(x_t, axis=0)
        return K.permute_dimensions(x_t, (1, 2, 3, 0))
        
        
    def call(self, x, training=None):
        #(w, h, channels) = x.get_shape()[1:]
        return self._process_input(x)
        #return x
    
    def compute_output_shape(self, input_shape):
        return input_shape