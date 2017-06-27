#!/usr/bin/python
# -*- coding: utf8 -*-
"""
This contains methods to load yolo/yolo2 configs and to convert it to valid keras network
"""

import re
from keras.layers import Input, Dense, Conv2D, Activation
from keras.layers.core import Dropout, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.local import LocallyConnected2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
import keras.backend as K



from keras.layers.normalization import BatchNormalization


def get_activation(activation):
    """
    Placeholder to handle different activation types
    """
    if activation == "leaky":
        return LeakyReLU(alpha=0.1)
    if activation == "logistic":
        return Activation(lambda x: 1.0/(1.0 + exp(-x)))
    return Activation(activation)


def read_config(filename):
    """
    returns collection of key-value pairs for each config file entry.
    """
    parseStr = lambda x: x.isalpha() and x or x.isdigit() and \
        int(x) or x[0] == '-' and x[1:].isdigit() and int(x) or \
        x.isalnum() and x != '0' and x or \
        len(set('.').intersection(x)) == 1 and \
        x.count('.') == 1 and float(x) or x != "0" and x or int(x)
        
    header = re.compile(r'\[(\S+)\]')
    comment = re.compile(r'\s*#.*')
    field = re.compile(r'(\S+)\s*=\s*(.+)')
    data = dict()
    last_class = None
    with open(filename) as config_file:
        for line in config_file:
            l = line.strip()
            if comment.match(l):
                continue
            header_match = header.match(l)
            if header_match:
                if len(data) > 0:
                    yield last_class, data
                    data = dict()
                last_class = header_match.groups()[0]
            field_match = field.match(l)
            if field_match:
                k, v = field_match.groups()
                v = list(map(lambda x: parseStr(x.strip()), v.split(',')))
                data[k] = v[0] if len(v) == 1 else v
        if len(data) > 0:
            yield last_class, data


def get_convolutional(params):
    from convolutional import Convolutional
    activation = get_activation(params.get('activation', 'linear'))
    batch_normalize = params.get('batch_normalize', 0) # TODO: add this param processing
    padding = "same" if params.get('pad', 0) else "valid"
    return Convolutional(**params)
    #return Conv2D(
    #    filters=params.get('filters', 1),
    #    kernel_size=params.get('size', 1),
    #    strides=params.get('stride', 1),
    #    padding=padding,
    #    activation=activation) 


def get_maxpool(params):
    return MaxPooling2D(
        strides=params.get('stride', 1), 
        pool_size=params.get('size', 1), 
        padding="same")


def get_local(params):
    activation = get_activation(params.get('activation', 'linear'))
    padding = "same" if params.get('pad', 0) else "valid"
    if padding != "valid":
        padding = "valid"
        #print("local with the same padding, should be fixed")
    return LocallyConnected2D(
        filters=params.get('filters', 1), 
        kernel_size=params.get('size', 1),
        strides=params.get('stride', 1),
        padding=padding,
        activation=activation)
    
    
def get_connected(params):
    from connected import Connected
    activation = get_activation(params.get('activation', "linear"))
    return Connected(**params)
    
    
def get_dropout(params):
    return Dropout(params.get('probability', 0.5))

def get_detection(params):
    from detection import Detection2D
    coords = params.get("coords", 1)
    classes = params.get("classes", 1)
    rescore = params.get("rescore", 0)
    num = params.get("num", 1)
    side = params.get("side", 7)
    object_scale = params.get("object_scale")
    noobject_scale = params.get("noobject_scale")
    class_scale = params.get("class_scale")
    coord_scale = params.get("coord_scale")
    #detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    softmax = params.get("softmax", 0)
    lsqrt = params.get("sqrt", 0)

    max_boxes = params.get("max", 30)
    coord_scale = params.get("coord_scale", 1)
    forced = params.get("forced", 0)
    object_scale = params.get("object_scale", 1)
    noobject_scale = params.get("noobject_scale", 1)
    class_scale = params.get("class_scale", 1)
    jitter = params.get("jitter", 0.2)
    random = params.get("random", 0)
    reorg = params.get("reorg", 0)
    #TODO: to be implemented
    return Detection2D(side, num, classes, coords, 
        object_scale, noobject_scale, class_scale, coord_scale)


def get_region(params):
    from region import Region
    return Region(**params)
    
layer_constructors = {
    'convolutional': get_convolutional,
    'maxpool': get_maxpool,
    'local': get_local,
    'connected': get_connected,
    'dropout': get_dropout,
    #'reorg'
    #'region'
    'detection': get_detection,
    'region': get_region
}


def buildYoloModel(config_filename):
    config_iterator = read_config(config_filename)
    _, net_params = config_iterator.next()
    h = net_params.get('height', 448)
    w = net_params.get('width', 448)
    c = net_params.get('channels', 3)
    lr = net_params.get('learning_rate')
    momentum = net_params.get('momentum')
    decay = net_params.get('decay')
    
    inputs = Input(shape=(h, w, c))
    outputs = inputs
    print(net_params)
    layer_names = [""]
    for class_name, params in config_iterator:
        layer = layer_constructors.get(class_name, lambda x: None)(params)
        if layer:
            outputs = layer(outputs)
            print("%15s : %s -> %s" % (
                class_name, 
                " x ".join(map(lambda x:"%4s"%x, layer.input_shape[1:])),
                " x ".join(map(lambda x:"%4s"%x, layer.output_shape[1:]))))
            layer_names.append(class_name)
        else:
            print(class_name, params)
        pass

    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay))
    # TODO: check what loss function and optimizer should be used here
    return model, layer_names


if __name__ == "__main__":
    # cfg/yolov1/yolo2.cfg
    buildYoloModel("cfg/yolov1/tiny-yolo.cfg")
