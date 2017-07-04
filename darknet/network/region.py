"""
There are two versions of detection layer in darknet, one of them is named `region`.
This module contains this layer implemented.
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


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
    
    
    def call(self, x, training=None):
        (w, h, channels) = x.get_shape()[1:]
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape