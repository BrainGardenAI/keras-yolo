# keras-yolo
Keras implementation of YOLO

## Current progress and usage
For now this package's functionality implementation is in progress. 

Current plan:

- [ ] add weights loading from darknet config (in progress)
- [ ] add script to compare run results vs darknet

Weights loading is more complex than it looks: there are several differences between keras and darknet, which should be carefully examined. For example, darknet allows batch normalization to be incorporated to layer, keras layers have no such an option.

To check currently implemented, run

```
python scripts/rebuild_and_load_weights.py
```

from the main project directory. This scripts expects to tiny-yolo.weights to be located at data folder, it also uses corresponding network config to create it and to load weights. But before that it builds necessary cython library with `distutils`. 

## Requirements

- keras
- distutils 

For development I use python 2, but this implementation is planned to be run with both python 2 & 3.
