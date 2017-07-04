"""Script to check darknet weights reading functionality.

Script runs distutils to rebuild sources and tries to read weights from darknet weights file.

This script should be run from the main project directory.
"""
import numpy as np
boxtype = np.dtype([('x', np.float32), ('y', np.float32), ('w', np.float32), ('h', np.float32)])

def rebuild_source():
    import subprocess, os, sys
    returncode = subprocess.call([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], cwd=os.getcwd())
    print("Finished rebuild with %s exit code" % returncode)
    return returncode



def check_weights_loading(model_file_name, weight_file_name):
    import os, sys
    sys.path.insert(0, os.getcwd())
    if not os.path.exists(model_file_name):
        print("darknet model config file %s not found" % model_file)
        return
    if not os.path.exists(weight_file_name):
        print("darknet weight file %s could not be found" % weight_file_name)
        return
    
    from darknet.weights_reader import read_file
    from darknet.network import buildYoloModel
    model, layer_data = buildYoloModel(model_file_name)
    print("-----\nstart actual reading\n-----\n")
    read_file(weight_file_name, model, layer_data)
    return model, layer_data


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
        new_h = (h*target_w)/w
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

def max_index(a, n): # float* and int
    if n <= 0:
        return -1
    i = 0
    max_i = 0
    max_v = a[0]
    for i in xrange(1, n):
        if a[i] > max_v:
            max_v = a[i]
            max_i = i
    return max_i


def draw_box_width(img, left, top, right, bottom, r, g, b):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom], outline=(r,g,b))
    #draw.text((10, 25-10), "world")

def draw_label(img, left, top, right, bottom, r, g, b, label):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    (w, h) = draw.textsize(label)
    draw.rectangle([left, top-h, left+w, top], fill=(r,g,b))
    draw.text((left, max(top-h, 0)), label, fill=(255,255,255))


def draw_detections(img, thresh, boxes, probs, names, alphabet, classes, w=13, h=13, n=5):
    for k in xrange(n):
        for j in xrange(h):
            for i in xrange(w):
                class0 = max_index(probs[i, j, k, :], classes)
                prob = probs[i, j, k, class0]
                if prob <= thresh:
                    continue
                # prob > thresh
                #width = h*0.012
                #printf("%s: %.0f%%\n", names[class0], prob*100);
                print("%s: %5i%%" % (names[class0], int(prob*100)))
                offset = class0*123457 % classes
                color = ((256*(class0+1)/(classes+1))+ (class0+1)/(classes+1))*256+(class0+1)/(classes+1)+offset
                red = color/256/256
                blue = color %256
                green = (color/256)%256
                #red = get_color(2,offset,classes);
                #green = get_color(1,offset,classes);
                #blue = get_color(0,offset,classes);
                b = boxes[i, j, k]
                left = (b["x"] - b["w"]/2.0)*img.width
                right = (b["x"] + b["w"]/2.0)*img.width
                top = (b["y"] - b["h"]/2.0)*img.height
                bottom = (b["y"] + b["h"]/2.0)*img.height
                left = max(left, 0)
                right = min(right, img.width - 1)
                top = max(top, 0)
                bottom = min(bottom, img.height - 1)
                draw_box_width(img, left, top, right, bottom, red, green, blue)
                draw_label(img, left, top, right, bottom, red, green, blue, names[class0])
    pass


def entry_index(n, entry, coords, classes):
    #simplified version - gets last coordinate in input tensor
    return n*(coords + classes + 1) + entry
    

def prepare_data(data, coords, background, classes, n):

    def softmax(data, temp=1):
        import numpy as np
        largest = np.max(data)
        data = np.exp(data/temp - largest/temp)
        s = sum(data)
        return data/s
    
    import numpy as np
    logistic_activate = lambda x: 1.0/(1.0 + np.exp(-x))
    (batches, w, h, channels) = data.shape
    for b in xrange(batches):
        for i in xrange(n):
            index = entry_index(i, 0, coords, classes)
            data[b, :, :, index] = logistic_activate(data[b, :, :, index])
            index = entry_index(i, 1, coords, classes)
            data[b, :, :, index] = logistic_activate(data[b, :, :, index])
            index = entry_index(i, 4, coords, classes)
            if not background:
                data[b, :, :, index] = logistic_activate(data[b, :, :, index])
    
    # here goes softmax_cpu
    #for b in xrange(batches*n):
    #    for g in xrange(w*h):
    #        #softmax(input + b*(inputs/n)+ g*1, classes, 1, w*h)
    for b in xrange(batches):
        index = entry_index(0, 5, coords, classes)
        for k in xrange(n):
            for i in xrange(w):
                for j in xrange(h):
                    data[b, i, j, index: index+classes] = softmax(data[b, i, j, index: index+classes])
            index += channels/n
    #int index = entry_index(l, 0, 0, l.coords + !l.background);
    #softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    return data


def get_region_box(features, biases, n, index, i, j, w, h, boxes):
    boxes[i, j, n]["x"] = (i + features[0, i, j, index])/w
    boxes[i, j, n]["y"] = (j + features[0, i, j, index + 1])/h
    boxes[i, j, n]["w"] = np.exp(features[0, i, j, index + 2])*biases[2*n]/w
    boxes[i, j, n]["h"] = np.exp(features[0, i, j, index + 3])*biases[2*n + 1]/h


def set_region_boxes(features, w, h, thresh, probs, boxes, oo, maps, hier_thresh, biases, 
        classes=80, coords=4, layer_w=13, layer_h=13, num=5):
    """
    sets probs and boxes values based on features and other parameters
    """
    features = np.swapaxes(features, 1, 2)
    for row in xrange(layer_h):
        for col in xrange(layer_w):
            for n in xrange(num): #n
                object_index = entry_index(n, 4, coords, classes)
                box_index = entry_index(n, 0, coords, classes)
                scale = features[0, col, row, object_index]
                get_region_box(features, biases, n, box_index, col, row, layer_w, layer_h, boxes)
                # sets boxes[col, row, box_index]
                if 1:
                    m = w if w > h else h
                    boxes[col, row, n]["x"] = (boxes[col, row, n]["x"] - (m-w)/2.0/m)/(float(w)/m)
                    boxes[col, row, n]["y"] = (boxes[col, row, n]["y"] - (m-h)/2.0/m)/(float(h)/m)
                    boxes[col, row, n]["w"] *= float(m)/w  
                    boxes[col, row, n]["h"] *= float(m)/h

                boxes[col, row, n]["x"] *= w
                boxes[col, row, n]["y"] *= h
                boxes[col, row, n]["w"] *= w 
                boxes[col, row, n]["h"] *= h
                class_index = entry_index(n, 5, coords, classes)
                m=0.0
                for j in xrange(classes):
                    class_index = entry_index(n, 5 + j, coords, classes) 
                    prob = scale*features[0, col, row, class_index]
                    probs[col, row, n, j] = prob if prob > thresh else 0.0
                    if prob > m: 
                        m = prob

                probs[col, row, n, classes] = m
                if oo:
                    probs[col, row, n, 0] = scale
    pass


def read_names(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()


def predict_image(model, layer_data, img_path):
    import PIL
    import numpy as np
    from darknet.network import do_nms_sort
    
    net_parameters = layer_data[0][1]
    output_parameters = layer_data[-1][1]
    
    classes = output_parameters.get("classes", 80)
    n = output_parameters.get("num", 5)
    coords = output_parameters.get("coords", 4)
    anchors = output_parameters.get("anchors", None)
    softmax = output_parameters.get("softmax", 0)
    #thresh = output_parameters.get("thresh", 0.24)
    thresh=0.24
    
    names = np.asarray(list(read_names("data/coco.names")))
    
    (im_w, im_h) = model.input_shape[1:-1]
    (w, h) = model.output_shape[1:-1]
    
    x = load_img(img_path, (im_w, im_h))
    img = PIL.Image.fromarray(np.uint8(x*255))
    #x=np.swapaxes(x, 0, 1)
    x = np.expand_dims(x, axis=0)
    #print(x[0,0,0,0], x[0,1,0,0],x[0,0,1,0], x[0,1,1,0], x.shape)
    features = model.predict(x)
    features = prepare_data(features, coords, 0, classes, n)
    
    boxes = np.zeros((w, h, n,), dtype=boxtype)
    probs = np.zeros((w, h, n, classes + 1), dtype=np.float32)
    biases = np.zeros((2*n,), dtype=np.float32)
    biases[:] = 0.5
    if anchors is not None:
        for i, x in enumerate(anchors):
            biases[i] = x
    
    hier_thresh=0.5

    print(features[0,0,0,0], features[0,1,0,0], features[0,0,1,0], features[0,0,0,1], features.shape)
    
    set_region_boxes(features, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh, 
            biases, classes=classes, coords=coords, layer_w=w, layer_h=h, num=n)
    nms=0.4
    
    if nms:
        do_nms_sort(boxes, probs, w*h*n, classes,  nms, w, h)
    draw_detections(img, thresh, boxes, probs, names, None, classes, w, h, n=n)
    img.save("predicted.jpg")
    print("images saved")
    
    
if __name__ == "__main__":
    resultcode = rebuild_source()
    if resultcode:
        print("Rebuild failed, exiting")
        exit(1)
    import keras.backend as K
    K.set_learning_phase(0)
    (model, layer_data) = check_weights_loading("../darknet/cfg/tiny_first.cfg", "data/tiny-yolo.weights")
    import sys
    if len(sys.argv) < 2:
        exit(0)
    predict_image(model, layer_data, sys.argv[1])
    #print(features[0,0,0])
