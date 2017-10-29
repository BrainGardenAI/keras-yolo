"""Script to check darknet weights reading functionality.

Script runs distutils to rebuild sources and tries to read weights from darknet weights file.

This script should be run from the main project directory.
"""
import argparse, sys
import numpy as np
#from builtins import range
import keras.backend as K


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
        print("darknet model config file %s not found" % model_file_name)
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
    for b in range(batches):
        for i in range(n):
            index = entry_index(i, 0, coords, classes)
            data[b, :, :, index] = logistic_activate(data[b, :, :, index])
            index = entry_index(i, 1, coords, classes)
            data[b, :, :, index] = logistic_activate(data[b, :, :, index])
            index = entry_index(i, 4, coords, classes)
            if not background:
                data[b, :, :, index] = logistic_activate(data[b, :, :, index])
    
    # here goes softmax_cpu
    #for b in range(batches*n):
    #    for g in range(w*h):
    #        #softmax(input + b*(inputs/n)+ g*1, classes, 1, w*h)
    for b in range(batches):
        index = entry_index(0, 5, coords, classes)
        for k in range(n):
            for i in range(w):
                for j in range(h):
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
    for row in range(layer_h):
        for col in range(layer_w):
            for n in range(num): #n
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
                for j in range(classes):
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


def predict_image(model, layer_data, img_path, class_names_file):
    import PIL
    import numpy as np
    from darknet.network import do_nms_sort
    from darknet.image_utils import load_img, draw_detections
    
    net_parameters = layer_data[0][1]
    output_parameters = layer_data[-1][1]
    
    classes = output_parameters.get("classes", 80)
    n = output_parameters.get("num", 5)
    coords = output_parameters.get("coords", 4)
    anchors = output_parameters.get("anchors", None)
    softmax = output_parameters.get("softmax", 0)
    #thresh = output_parameters.get("thresh", 0.24)
    thresh=0.24
    
    names = np.asarray(list(read_names(class_names_file)))
    
    (im_w, im_h) = model.input_shape[1:-1]
    (w, h) = model.output_shape[1:-1]
    
    x = load_img(img_path, (im_w, im_h))
    img = PIL.Image.fromarray(np.uint8(x*255))
    #x=np.swapaxes(x, 0, 1)
    x = np.expand_dims(x, axis=0)
    #print(x[0,0,0,0], x[0,1,0,0],x[0,0,1,0], x[0,1,1,0], x.shape)
    features = model.predict(x)
    #features = prepare_data(features, coords, 0, classes, n)
    #features = features.reshape((-1, w, h, n*(classes+coords+1)))
    print(features.shape)
    boxes = np.zeros((w, h, n,), dtype=boxtype)
    probs = np.zeros((w, h, n, classes + 1), dtype=np.float32)
    biases = np.zeros((2*n,), dtype=np.float32)
    biases[:] = 0.5
    if anchors is not None:
        for i, x in enumerate(anchors):
            biases[i] = x
    
    hier_thresh=0.5
    print(features.shape)
    #features = features.reshape((-1, 7,7,30))

    print(features[0,0,0,0], features[0,1,0,0], features[0,0,1,0], features[0,0,0,1], features.shape)
    
    set_region_boxes(features, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh, 
            biases, classes=classes, coords=coords, layer_w=w, layer_h=h, num=n)
    nms=0.4
    
    if nms:
        do_nms_sort(boxes, probs, w*h*n, classes,  nms, w, h)
    draw_detections(img, thresh, boxes, probs, names, None, classes, w, h, n=n)
    img.save("predicted.jpg")
    print("images saved")


def parse_args():
    parser = argparse.ArgumentParser(description='Keras-YOLO run script')
    parser.add_argument('--cfg_file',
        default="../darknet/cfg/tiny_first.cfg",
        help='darknet .cfg file')
    parser.add_argument('--weights_file',
        default="data/tiny-yolo.weights",
        help='darknet .weights file')
    parser.add_argument('image_file',
        help='image file to process')
    parser.add_argument("--class_names_file",
        default="data/coco.names",
        help="file with list of class names - 1 at each line")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resultcode = rebuild_source()
    if resultcode:
        print("Rebuild failed, exiting")
        exit(1)
    K.set_learning_phase(0)
    (model, layer_data) = check_weights_loading(args.cfg_file, args.weights_file)
    if len(sys.argv) < 2:
        exit(0)
    predict_image(model, layer_data, args.image_file, args.class_names_file)
    #print(features[0,0,0])
