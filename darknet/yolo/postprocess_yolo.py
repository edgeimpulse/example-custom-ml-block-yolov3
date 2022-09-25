import torch
import numpy as np
import collections
import cv2

import struct
from PIL import Image
import sys

"""Anchors"""
anchors_yolov3 = [10, 13, 16, 30, 33, 23, 30, 61,  62,45,  59,119,  116,90,  156,198,  373,326]
anchors_yolov2 = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchors_tinyyolov3 = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
anchors_tinyyolov2 = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

"""Labels"""
#Label for YOLOv2/Tiny YOLOv2
labels_voc = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

"""Output Layer Number"""
out_layer_num = 3 # Default is YOLOv3


def predict_transform_yolov3(preds, inp_size, _anchors, num_classes):
    N, C, H, W = preds.shape
    grid_size = H
    anchors = _anchors
    grid_num = inp_size // grid_size
    num_anchors = len(anchors)
    bbox_attrs = num_classes + 5 # x,y,w,h + prob + class0_prob,class1_prob...

    preds = preds.view(N, bbox_attrs*num_anchors, grid_size*grid_size) #(N, (5+num_classes)*n_anchors, H*W)
    preds = preds.permute(0, 2, 1).contiguous()                        #N, (H*W), (5+num_classes)*n_anchors
    preds = preds.view(N, grid_size*grid_size*num_anchors,  bbox_attrs) #N, (H*W)*n_anchors, (5+num_classes)

    # X, Y
    grid = np.arange(grid_size)
    x_offsets, y_offsets = np.meshgrid(grid, grid)

    x_offsets = torch.tensor(x_offsets, dtype=torch.float).view(-1, 1)
    y_offsets = torch.tensor(y_offsets, dtype=torch.float).view(-1, 1)
    x_y_offsets = torch.cat([x_offsets, y_offsets], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    preds[..., :2] = torch.sigmoid(preds[..., :2])	# center
    preds[..., :2] += x_y_offsets #at this stage,(bx = sigmoid(tx) + cx ) and (by = sigmoid(ty) + cy) is done
    preds[..., :2] /= grid_size

    # objectness
    obj_probs = torch.sigmoid(preds[..., 4])

    # W, H
    anchors = torch.tensor(anchors, dtype=torch.float)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    preds[..., 2:4] = torch.exp(preds[..., 2:4])*anchors / inp_size

    # class probability
    class_scores = torch.sigmoid(preds[..., 5:])

    #correct_yolo_boxes
    w = inp_size
    h = inp_size
    if inp_size/w < inp_size/h :
        new_w = inp_size
        new_h = (h * inp_size)/w
    else :
        new_h = inp_size
        new_w = (w * inp_size)/h

    preds[...,0] = (preds[...,0] - (inp_size - new_w)/2./inp_size) /(new_w/inp_size)
    preds[...,1] = (preds[...,1] - (inp_size - new_h)/2./inp_size) /(new_h/inp_size)
    preds[...,2] = preds[...,2] * inp_size/new_w
    preds[...,3] = preds[...,3] * inp_size/new_h
    boxes = preds[..., :4] * inp_size
    return boxes, obj_probs, class_scores #(center_x, center_y, width, height)

def predict_transform_yolov2(preds, inp_size, _anchors, num_classes):
    N, C, H, W = preds.shape
    grid_size = H
    anchors = _anchors
    grid_num = inp_size // grid_size
    num_anchors = len(anchors)
    bbox_attrs = num_classes + 5 # x,y,w,h + prob + class0_prob,class1_prob...

    preds = preds.view(N, bbox_attrs*num_anchors, grid_size*grid_size) #(N, (5+num_classes)*n_anchors, H*W)
    preds = preds.permute(0, 2, 1).contiguous()                        #N, (H*W), (5+num_classes)*n_anchors
    preds = preds.view(N, grid_size*grid_size*num_anchors,  bbox_attrs) #N, (H*W)*n_anchors, (5+num_classes)

    # X, Y
    grid = np.arange(grid_size)
    x_offsets, y_offsets = np.meshgrid(grid, grid)

    x_offsets = torch.tensor(x_offsets, dtype=torch.float).view(-1, 1)
    y_offsets = torch.tensor(y_offsets, dtype=torch.float).view(-1, 1)
    x_y_offsets = torch.cat([x_offsets, y_offsets], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    preds[..., :2] = torch.sigmoid(preds[..., :2])	# center
    preds[..., :2] += x_y_offsets #at this stage, (bx = sigmoid(tx) + cx ) and (by = sigmoid(ty) + cy) is done

    # objectness
    obj_probs = torch.sigmoid(preds[..., 4])

    # W, H
    anchors = torch.tensor(anchors, dtype=torch.float)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    preds[..., 2:4] = torch.exp(preds[..., 2:4])*anchors

    # class probability
    class_scores = preds[..., 5:]

    #correct_region_boxes
    w = inp_size
    h = inp_size
    if inp_size/w < inp_size/h :
        new_w = inp_size
        new_h = (h * inp_size)/w
    else :
        new_h = inp_size
        new_w = (w * inp_size)/h

    preds[...,0] = (preds[...,0] - (inp_size - new_w)/2./inp_size) /(new_w/inp_size)
    preds[...,1] = (preds[...,1] - (inp_size - new_h)/2./inp_size) /(new_h/inp_size)
    preds[...,2] = preds[...,2] * inp_size/new_w
    preds[...,3] = preds[...,3] * inp_size/new_h
    boxes = preds[..., :4] * grid_num
    return boxes, obj_probs, torch.softmax(class_scores, -1) #(center_x, center_y, width, height)


def all_post_process(model_name, preds, anchors, input_size, grid_size, n_classes, obj_th, nms_th):
    boxes = {}
    obj_probs = {}
    class_scores = {}

    # Model unique configuration
    if model_name == "yolov3":
        _anchors = np.array(anchors).reshape(-1, 3, 2)
        _anchors = _anchors[[2, 1, 0]]
        predict_transform = predict_transform_yolov3
    elif model_name=="tinyyolov3":
        _anchors = np.array(anchors).reshape(-1, 3, 2)
        _anchors = _anchors[[1, 0]]
        predict_transform = predict_transform_yolov3
    elif model_name in ["yolov2", "tinyyolov2"]:
        _anchors = np.array(anchors).reshape(-1, 5, 2)
        predict_transform = predict_transform_yolov2
    _anchors = _anchors.tolist()

    for i in range(out_layer_num):
        # NN Output -> box coordinate, score
        boxes[i], obj_probs[i], class_scores[i] = predict_transform(preds[i], input_size, _anchors[i], n_classes)
        # object score threshold
        boxes[i], obj_probs[i], class_scores[i] = threshold_predictions(boxes[i], obj_probs[i], class_scores[i], th=obj_th)
        # box coordinate(centerX, centerY,) to top_left_x, top_left_y, bottom_right_x, bottom_right_y
        boxes[i] = centerwh_to_corners(boxes[i])
    # Non-Maximum Suppression (NMS)
    detections = nms(boxes, obj_probs, class_scores, th=nms_th)

    return detections

def threshold_predictions(boxes, obj_probs, class_scores, th=0.6):
    boxes.squeeze_(0)
    class_scores.squeeze_(0)
    obj_probs = obj_probs.view(class_scores.shape[0], 1)

    box_scores = class_scores * obj_probs
    box_class_scores, box_classes = box_scores.max(1)
    obj_probs = obj_probs.squeeze(1)

    filtering_mask = box_class_scores >= th

    boxes = boxes[filtering_mask]
    scores = box_class_scores[filtering_mask]
    classes = box_classes[filtering_mask]

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()

    return boxes, scores, classes

def nms(_boxes, _scores, _classes, th):
    detections = collections.defaultdict(lambda: [])
    boxes = _boxes[0]
    scores = _scores[0]
    classes = _classes[0]
    for i in range(out_layer_num -1):
        boxes = np.concatenate([boxes, _boxes[i+1]])
        scores = np.append(scores, _scores[i+1])
        classes = np.append(classes, _classes[i+1])
    detected_classes = np.unique(classes)
    for cls in detected_classes:
        class_mask = classes == cls
        cls_boxes = boxes[class_mask]
        cls_scores = scores[class_mask]

        sort_idxs = np.argsort(-cls_scores, 0)
        cls_boxes = cls_boxes[sort_idxs]
        cls_scores = cls_scores[sort_idxs]
        while 1 < cls_boxes.shape[0]:
            detections[cls].append((cls_boxes[0], cls_scores[0]))
            ious = _iou_np(cls_boxes[0], cls_boxes[1:])
            iou_mask = ious < th
            cls_boxes = cls_boxes[1:][iou_mask]
            cls_scores = cls_scores[1:][iou_mask]

        if cls_boxes.shape[0] == 1:
            detections[cls].append((cls_boxes[0], cls_scores[0]))

    return detections

def draw_predictions(im, detections, labels):
    for i, label in enumerate(labels):
        for (box, score) in detections[i]:
            print('Class: {} | Probability {:5.1%} | [X1, Y1, X2, Y2] = [{:.0f},{:.0f},{:.0f},{:.0f}]'.format(label, score, box[0],box[1],box[2],box[3]))

            box = box.astype(np.int32)
            im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 1)
            im = cv2.rectangle(im, (box[0] + 1, box[1] + 1), (box[2] - 1, box[3] - 1), (0, 0, 0), 1)

            c1 = tuple(box[:2])
            fontface = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 0.4
            thickness = 1
            textRectColor = [45, 116, 229]  # BGR
            textColor = [255, 255, 255]
            caption = "{0}:{1:5.1%}".format(label,score)
            t_size = cv2.getTextSize(caption, fontface, fontScale, thickness)[0]
            c2 = c1[0] + t_size[0] + 2, c1[1] + t_size[1] + 5
            cv2.rectangle(im, (c1[0], c1[1]), (c2[0] + 2, c2[1] + 2), textRectColor, -1)
            cv2.putText(im, caption, (int(box[0]), int(box[1]) + t_size[1] + 4), fontface, fontScale, textColor, thickness)
    return im

def centerwh_to_corners(boxes):
    if len(boxes) == 0:
        return boxes

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    boxes = boxes.reshape(-1, 4)
    return np.concatenate([boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2], 1)

def _iou_np(box, others):
    box, others = box.reshape(-1, 4), others.reshape(-1, 4)
    box = box.repeat(repeats=others.shape[0], axis=0)
    x1y1 = np.maximum(box[:, :2], others[:, :2])
    x2y2 = np.minimum(box[:, 2:], others[:, 2:])
    inter_edges = np.maximum(x2y2 - x1y1, 0)
    inter_areas = inter_edges[:, 0] * inter_edges[:, 1]

    box_edges = box[:, 2:4] - box[:, :2]
    box_areas = box_edges[:, 0] * box_edges[:, 1]

    others_edges = others[:, 2:4] - others[:, :2]
    others_areas = others_edges[:, 0] * others_edges[:, 1]

    union_areas = box_areas + others_areas - inter_areas

    return inter_areas / union_areas


if __name__ == '__main__':
    if ( len(sys.argv) > 1 and sys.argv[1] in ["yolov3", "yolov2", "tinyyolov3", "tinyyolov2"]):
        model_name = sys.argv[1]
        print("Model [", model_name, "] is selected.")
    else:
        print("Default model [ yolov3 ] is selected.")
        model_name = "yolov3"

    print('whu')
    exit(1)

    # Load label list for YOLOv3/Tiny YOLOv3
    f=open("coco-labels-2014_2017.txt")
    labels = f.read().splitlines()
    f.close()

    output_shape = []
    model_in_size = 416

    if model_name=="yolov3":
        anchors = anchors_yolov3
        out_layer_num = 3
        num_class = len(labels)
        output_shape.append([1, 3*(5+num_class), 13, 13]) # [N, C, H, W]
        output_shape.append([1, 3*(5+num_class), 26, 26]) # [N, C, H, W]
        output_shape.append([1, 3*(5+num_class), 52, 52]) # [N, C, H, W]
    elif model_name=="yolov2":
        anchors = anchors_yolov2
        labels = labels_voc
        num_class = len(labels)
        out_layer_num = 1
        output_shape.append([1, 5*(5+num_class), 13, 13]) # [N, C, H, W]
    elif model_name=="tinyyolov3":
        anchors = anchors_tinyyolov3
        out_layer_num = 2
        num_class = len(labels)
        output_shape.append([1, 3*(5+num_class), 13, 13]) # [N, C, H, W]
        output_shape.append([1, 3*(5+num_class), 26, 26]) # [N, C, H, W]
    elif model_name=="tinyyolov2":
        anchors = anchors_tinyyolov2
        labels = labels_voc
        num_class = len(labels)
        out_layer_num = 1
        output_shape.append([1, 5*(5+num_class), 13, 13]) # [N, C, H, W]

    result_bin = open("sample.bmp.bin", 'rb')

    data = []
    for n in range(out_layer_num):
        data.append(np.zeros(output_shape[n], dtype=float))

    # Load DRP-AI output binary
    for n in range(out_layer_num): # Output number
        for c in range(output_shape[n][1]): # C
            for h in range(output_shape[n][2]): # H
                for w in range(output_shape[n][3]): # W
                    a = struct.unpack('<f', result_bin.read(4))
                    data[n][0, c, h, w] = a[0]

    # Read image to draw bounding boxes
    im = Image.open("sample.bmp")
    im = im.resize((model_in_size, model_in_size), resample=Image.BILINEAR)

    # Post-processing
    preds = {}
    for i in range(out_layer_num):
        preds[i] = torch.from_numpy(data[i]).clone()
    detections = all_post_process(model_name, preds, anchors, input_size=model_in_size,
        grid_size=output_shape, n_classes=num_class, obj_th=0.5, nms_th=0.5)

    # Draw bounding box
    img = np.asarray(im) # RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# BGR
    ret_im = draw_predictions(img, detections, labels) # BGR
    cv2.imwrite("result.jpg", ret_im)
