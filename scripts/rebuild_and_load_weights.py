"""Script to check darknet weights reading functionality.

Script runs distutils to rebuild sources and tries to read weights from darknet weights file.

This script should be run from the main project directory.
"""

def rebuild_source():
    import subprocess, os, sys
    returncode = subprocess.call([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], cwd=os.getcwd())
    print("Finished rebuild with %s exit code" % returncode)
    return returncode



def check_weights_loading(model_file_name, weight_file_name):
    import os, sys
    #if not os.path.exists(model_file_name):
    #    print("darknet model config file %s not found" % model_file)
    #    return
    if not os.path.exists(weight_file_name):
        print("darknet weight file %s could not be found" % weight_file_name)
        return
    sys.path.insert(0, os.getcwd())
    from darknet.weights_reader import read_file
    from darknet.network import buildYoloModel
    model, layer_names = buildYoloModel(model_file_name)
    #print([(x.shape)  for x in model.get_weights()])
    for layer_type, layer in zip(layer_names, model.layers):
        print(layer_type, layer.name)
        print [ x.shape for x in layer.get_weights()]
        print(layer.output_shape)
    print("-----\nstart actual reading\n-----\n")
    read_file(weight_file_name, model, layer_names)
    return model

def resize_image(data, size):
    """
    resizes given image (as an array in which channels are given in last dimension) to given size
    """
    import numpy as np
    import scipy
    (w, h, channels) = data.shape
    (target_w, target_h) = size
    scale_w = 1.0*(w - 1)/(target_w - 1)
    scale_h = 1.0*(h - 1)/(target_h - 1)
    part = np.zeros((target_w, h, 3))
    for c in xrange(channels):
        arr = np.asarray([[ min(i*scale_w, w - 1) for i in xrange(target_w)]])
        for j in xrange(h):
            part[:, j, c] = scipy.ndimage.map_coordinates(data[:, j, c], arr, order=1)
    resized = np.zeros((target_w, target_h, channels))
    for c in xrange(channels):
        arr = np.asarray([[min(j*scale_h, h - 1) for j in xrange(target_h)]])
        for i in xrange(target_w):
            resized[i, :, c] = scipy.ndimage.map_coordinates(part[i, :, c], arr, order=1)
    return resized
    
    
def letterbox_image(img_data, size):
    """
    exactly the same as letterbox_image from image.c from darknet sources:
    resizes image (given as a RGB 0-1 float values) and fills empty space with 0.5
    """
    import numpy as np
    import scipy
    (target_w, target_h) = size
    new_w = target_w
    new_h = target_h
    (w, h, c) = img_data.shape
    if 1.0*target_w/w < 1.0*target_h/h:
        new_w = target_w
        new_h = (h*target_w)/h
    else:
        new_h = target_h
        new_w = (w*target_h)/h
    resized = resize_image(img_data, (new_w, new_h))
    assert resized.shape == (new_w, new_h, c)
    x0, y0 = (target_w - new_w)/2, (target_h - new_h)/2 # upper left corner of resized image
    resized = np.pad(
        resized, 
        [
            (max(x0 - 1, 0), max(target_w - new_w - max(x0 - 1, 0), 0)), 
            (max(y0 - 1, 0), max(target_h - new_h - max(y0 - 1, 0), 0)), 
            (0, 0)
        ], 'constant', constant_values=0.5)
    return resized
 

def load_img(img_path, target_size):
    """
    loads the image the same way darknet does, processes it and returns it (as array).
    uses PIL, like keras.preprocessing.image module.
    This loads image in RGB format.
    """
    from PIL import Image as pil_image
    import keras.backend as K
    import numpy as np
    img = pil_image.open(img_path)
    # TODO: check format and convert to RGB
    #resize
    x = np.asarray(img, dtype=K.floatx())/255.0
    #print(x[0,0,0], x[1,0,0], x[0,1,0], x[1,1,0], img.mode)
    x = letterbox_image(x, target_size)
    return x
    
    
def predict_image(model, img_path):
    #print(model.input_shape[1: -1])
    import numpy as np
    x = load_img(img_path, model.input_shape[1:-1])
    x = np.expand_dims(x, axis=0)
    #print(x[0,0,0,0], x[0,1,0,0],x[0,0,1,0], x[0,1,1,0], x.shape)
    features = model.predict(x)
    print(features[0,0,0,0], features[0,1,0,0], features[0,0,1,0], features[0,0,0,1], features.shape)
    pass
    
    
if __name__ == "__main__":
    resultcode = rebuild_source()
    if resultcode:
        print("Rebuild failed, exiting")
        exit(1)
    import keras.backend as K
    K.set_learning_phase(0)
    model = check_weights_loading("../darknet/cfg/tiny-yolo.cfg", "data/tiny-yolo.weights")
    import sys
    if len(sys.argv) < 2:
        exit(0)
    predict_image(model, sys.argv[1])
    #print(features[0,0,0])
