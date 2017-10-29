#from builtins import range
import numpy as np
cimport numpy as np


cdef extern from "box.h":
    ctypedef struct box:
        float x, y, w, h
        
    float box_iou(box a, box b)


def box_iou_(a, b):
    return box_iou(
        box(a['x'], a['y'], a['w'], a['h']), 
        box(b['x'], b['y'], b['w'], b['h']))
       

#cpdef do_nms_sort2(np.ndarray boxes, 
#      np.ndarray probs, int total, 
#      int classes, np.float32_t thresh):
#    #print(probs, total, classes)
#    box.do_nms_sort(<box.boxtype*>&boxes.data, <float*>&probs.data, total, classes, thresh)

cpdef do_nms_sort(np.ndarray boxes, np.ndarray probs, int total, int classes, 
        np.float32_t thresh, int w, int h):
        
    cdef int i, j, k
    cdef np.ndarray s = np.zeros((total,), dtype=np.int32)
    s[:] = 0
    boxes_flatten = boxes.flatten()
    for k in range(classes):
        s[:] = k
        probs_k = probs[:, :, :, k].flatten()
        sorted_indices = np.argsort(probs_k, axis=None)[::-1]
        for i in range(total):
            if probs_k[sorted_indices[i]] == 0: 
                continue
            a = boxes_flatten[sorted_indices[i]]
            j = i + 1
            while j < total:
                b = boxes_flatten[sorted_indices[j]]
                if box_iou(a, b) > thresh:
                    r = sorted_indices[j]
                    n = r // (w*h)
                    r = r - n*w*h
                    c = r//w
                    r = r - c*w
                    probs[r, c, n] = 0
                j += 1
