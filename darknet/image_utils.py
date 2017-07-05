import numpy as np
boxtype = np.dtype([('x', np.float32), ('y', np.float32), ('w', np.float32), ('h', np.float32)])


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