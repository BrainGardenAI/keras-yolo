from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Detection2D(Layer):
    """
    Detection layer for 2D inputs
    """
    def __init__(self, side, num, classes, coords, object_scale, 
            noobject_scale, class_scale, coord_scale, **kwargs):
        data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format must be in '
                             '{"channels_last", "channels_first"}')
        self.data_format = data_format
        self.side = side 
        self.n = num
        self.classes = classes
        self.coords = coords
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        super(Detection2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # add check to ensure input_shape size is the same as expected
        ll = np.prod(filter(lambda x: x is not None, input_shape))
        # print(input_shape, self.side*self.side*((1 + self.coords)*self.n + self.classes))
        assert(self.side*self.side*((1 + self.coords)*self.n + self.classes) == ll)
        super(Detection2D, self).build(input_shape)

    def call(self, x, training=None):
        return x #K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        print(input_shape)
        if self.data_format == 'channels_first':
            return (None, self.side, self.side, 
                (1 + self.coords)*self.n + self.classes)
        return (self.side, self.side, 
            (1 + self.coords)*self.n + self.classes, None)
    
    def get_config(self):
        config = {'side': self.side,
                  'n': self.n,
                  'classes': self.classes,
                  'coords': self.coords,
                  'object_scale': self.object_scale,
                  'noobject_scale': self.noobject_scale,
                  'class_scale': self.class_scale,
                  'coord_scale': self.coord_scale}
        base_config = super(Detection2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
