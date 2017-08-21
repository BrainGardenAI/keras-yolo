# keras-yolo
Keras implementation of [YOLO - real-time object detection system](https://pjreddie.com/darknet/yolo/).

## Current progress and usage
For now this package's functionality implementation is in progress. 

Current plan:

- [x] add weights loading from darknet config (in progress)
- [?] add script to compare run results vs darknet (currently I compare results with darknet manually).

Weights loading is more complex than it looks: there are several differences between keras and darknet, which should be carefully examined. For example, darknet allows batch normalization to be incorporated to layer, keras layers have no such an option and provides distinct `BatchNormalization` class instead.

To check currently implemented, run

```
python scripts/rebuild_and_load_weights.py
```

from the main project directory. This scripts expects to tiny-yolo.weights to be located at data folder, it also uses corresponding network config to create it and to load weights. But before that it builds necessary cython library with `distutils`. 

If you run 
```
python scripts/rebuild_and_load_weights.py ../darknet/dog.jpg
```
it tries to load image from given path and to predict its class (this is in progress). 

As a result of this command, file `predicted.jpg` will be generated and it will look similar to this:

![Generated image with detections](https://github.com/BrainsGarden/keras-yolo/blob/master/predicted.jpg)

There still remains a pythonish problem with precision - that's why detection borders 
at resulting image slightly differ from what you see when you run `darknet`.

## Requirements

- keras
- distutils 
- cython

For development I use python 2, but this implementation is planned to be run with both python 2 & 3.
