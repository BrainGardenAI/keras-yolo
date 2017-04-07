#!/usr/bin/python
# -*- coding: utf8 -*-
"""
This contains methods to load yolo/yolo2 configs and to convert it to valid keras network
"""

import re
from keras.layers import Input, Dense, Conv2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.local import LocallyConnected2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD


def get_activation(activation):
    """
    Placeholder to handle different activation types
    """
    if activation == "leaky":
        return LeakyReLU(alpha=0.1)
    return activation


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
    activation = get_activation(params.get('activation', 'linear'))
    batch_normalize = params.get('batch_normalize') # TODO: add this param processing
    padding = "same" if params.get('pad', 0) else "valid"
    return Conv2D(
        filters=params.get('filters', 1),
        kernel_size=params.get('size', 1),
        strides=params.get('stride', 1),
        padding=padding,
        activation=activation) 


def get_maxpool(params):
    return MaxPooling2D(
        strides=params.get('stride', 1), 
        pool_size=params.get('size', 1))


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
    activation = get_activation(params.get('activation', "linear"))
    return Dense(
        params.get('output', 1),
        activation=activation
    )

def get_dropout(params):
    return Dropout(params.get('probability', 0.5))

def get_detection(params):
    pass


layer_constructors = {
    'convolutional': get_convolutional,
    'maxpool': get_maxpool,
    'local': get_local,
    'connected': get_connected,
    'dropout': get_dropout,
    #'route'
    #'reorg'
    #'region'
    'detection': get_detection
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

    for class_name, params in config_iterator:
        layer = layer_constructors.get(class_name, lambda x: None)(params)
        if layer:
            outputs = layer(outputs)
        else:
            print(class_name, params)
        pass

    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay))
    # TODO: check what loss function and optimizer should be used here
    return model


if __name__ == "__main__":
    buildYoloModel("cfg/yolov1/yolo2.cfg")
