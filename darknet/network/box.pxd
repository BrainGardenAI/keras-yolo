
import numpy as np
cimport numpy as np

cdef extern from "box.h":
    ctypedef struct box:
        float x, y, w, h
        
    float box_iou(box a, box b)
    #void do_nms_sort(boxtype* boxes, float *probs, int total, int classes, float thresh)