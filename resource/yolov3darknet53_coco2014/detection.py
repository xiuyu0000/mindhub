import os
import sys
from collections import defaultdict

import cv2
import random
import easydict
import numpy as np

DEFAULT_CONFIG = easydict.EasyDict({
    "num_classes": 80,
    "infer_ignore_threshold": 0.25,
    "nms_thresh": 0.65,
})


def _nms(predicts, threshold):
    """Calculate NMS."""
    x1 = predicts[:, 0]
    y1 = predicts[:, 1]
    x2 = x1 + predicts[:, 2]
    y2 = y1 + predicts[:, 3]
    scores = predicts[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_area = intersect_w * intersect_h
        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

        indexes = np.where(ovr <= threshold)[0]
        order = order[indexes + 1]
    return reserved_boxes


def _draw_label_img(draw_img, bbox, label, score):
    x_l, y_t, w, h = bbox
    x_r, y_b = x_l + w, y_t + h
    x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
    _color = [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(draw_img, (x_l, y_t), (x_r, y_b), _color, 2)

    text = f"{label}: {score.round(2)}"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(draw_img, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(_color), -1)
    cv2.putText(draw_img, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return draw_img


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args=None):
        if not args:
            args = DEFAULT_CONFIG
        self.infer_ignore_threshold = args.infer_ignore_threshold
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                       'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                       'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                       'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.num_classes = args.num_classes
        self.results = defaultdict(list)
        self.file_path = ''
        self.det_boxes = []
        self.nms_thresh = args.nms_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for clsi in self.results:
            dets = self.results[clsi]
            dets = np.array(dets)
            keep_index = _nms(dets, self.nms_thresh)

            keep_box = [{'category': self.labels[int(clsi)],
                         'bbox': list(dets[i][:4].astype(float)),
                         'score': dets[i][4].astype(float)}
                        for i in keep_index]
            self.det_boxes.extend(keep_box)

    def detect(self, outputs, image_shape):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|1, 52, 52, 3, 85| ]
        for out_id in range(outputs_num):
            # 1, 52, 52, 3, 85
            out_item_single = outputs[out_id]
            # get number of items in one head, [B, gx, gy, anchors, 5+80]
            dimensions = out_item_single.shape[:-1]
            out_num = 1
            for d in dimensions:
                out_num *= d
            ori_w, ori_h = image_shape
            # img_id = int(image_id[batch_id])
            x = out_item_single[..., 0] * ori_w
            y = out_item_single[..., 1] * ori_h
            w = out_item_single[..., 2] * ori_w
            h = out_item_single[..., 3] * ori_h

            conf = out_item_single[..., 4:5]
            cls_emb = out_item_single[..., 5:]

            cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
            x = x.reshape(-1)
            y = y.reshape(-1)
            w = w.reshape(-1)
            h = h.reshape(-1)
            cls_emb = cls_emb.reshape(-1, self.num_classes)
            conf = conf.reshape(-1)
            cls_argmax = cls_argmax.reshape(-1)

            x_top_left = x - w / 2.
            y_top_left = y - h / 2.
            # create all False
            flag = np.random.random(cls_emb.shape) > sys.maxsize
            for i in range(flag.shape[0]):
                c = cls_argmax[i]
                flag[i, c] = True
            confidence = cls_emb[flag] * conf
            for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                if confi < self.infer_ignore_threshold:
                    continue
                x_lefti = max(0, x_lefti)
                y_lefti = max(0, y_lefti)
                wi = min(wi, ori_w)
                hi = min(hi, ori_h)

                self.results[clsi].append([x_lefti, y_lefti, wi, hi, confi])

    def save_bbox_img(self, img, save_path):
        """Show image."""
        for dbox in self.det_boxes:
            bbox, label, score = dbox["bbox"], dbox["category"], dbox["score"]
            img = _draw_label_img(img, bbox, label, score)

        cv2.imwrite(save_path, img)
