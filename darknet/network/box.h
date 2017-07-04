/** box.h + box.c were taken from original darknet repo at
https://github.com/pjreddie/darknet.

Currently only `box_iou` function is in use, `do_nms_sort` was rewritten in cython.
I plan to use this code during learning phase if necessary, that's why currently 
it contains full box.c+box.h sources.
*/
#ifndef BOX_H
#define BOX_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f, int stride);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float *probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float *probs, int total, int classes, float thresh);
void do_nms_obj(box *boxes, float *probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
