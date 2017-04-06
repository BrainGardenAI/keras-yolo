#!/usr/bin/python
# -*- coding: utf8 -*-
"""
This contains methods to load yolo/yolo2 configs and to convert it to valid keras network
"""

import re
from keras.layers import Input, Dense, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.local import LocallyConnected2D


def read_config(filename):
    """
    returns collection of key-value pairs for each config file entry.
    """
    parseStr = lambda x: x.isalpha() and x or x.isdigit() and \
        int(x) or x[0] == '-' and x[1:].isdigit() and int(x) or \
        x.isalnum() and x or len(set('.').intersection(x)) == 1 and \
        x.count('.') == 1 and float(x) or x
        
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


def get_convolutional(params):
    return Conv2D(
        filters=params.get('filters', 1),
        kernel_size=params.get('size', 1),
        strides=params.get('stride', 1),
        padding="same") 


def get_maxpool(params):
    return MaxPooling2D(
        strides=params.get('stride', 1), 
        pool_size=params.get('size', 1), 
        padding="same")
    
    
layer_constructors = {
    'convolutional': get_convolutional,
    'maxpool': get_maxpool,
    #'dropout': get_dropout,
    #'detection': get_detection
    #'route'
    #'reorg'
    #'region'
}


config_iterator = read_config("cfg/yolov1/yolo2.cfg")
_, net_params = config_iterator.next()
h = net_params.get('height', 416)
w = net_params.get('width', 416)
c = net_params.get('channels', 416)
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


Model(inputs=inputs, outputs=outputs)
