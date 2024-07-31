import torch
import torchvision
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Union
from PIL import Image
import random

### Return pretrained model, model weights, pretrain data classes, and model transforms ###
def get_model_weights_classes_transforms():
    # load a model pre-trained on COCO
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    classes = weights.meta["categories"] # list of strings
    transforms = weights.transforms()

    return model, weights, classes, transforms



def filter_boxes_labels_scores(boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, min_score: float=0.6):
    """
    Returns a filtered selection of boxes, labels, and scores based on the minimum score threshold `min_score` which falls on the range [0.0, 1.0]
    """

    filtered_idxs = np.arange(len(scores))
    filtered_idxs = [idx for idx in filtered_idxs if scores[idx] >= min_score]

    filtered_boxes = boxes[filtered_idxs]
    filtered_labels = labels[filtered_idxs]
    filtered_scores = scores[filtered_idxs]

    return filtered_boxes, filtered_labels, filtered_scores



def draw_boxes_on_frame(frame, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, classes: List[str]):
    """
  Parameters:
    frames
        A frame

    boxes: torch.Tensor, shape (N, 4)
        box coordinates

    labels: torch.Tensor, shape (N,)
        direct numerical outputs of model

    scores: torch.Tensor, shape(N,)
        scores for each object detected in image

    classes: List[str]
        list of classes

    """

    for box, label, score in zip(boxes, labels, scores):
        start_x, start_y = int(box[0]), int(box[1])
        end_x, end_y = int(box[2]), int(box[3])
            
        color = tuple(random.randint(0, 255) for _ in range(3))
        frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        frame = cv2.putText(frame, f"{classes[label]} {score:.2f}", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    return frame


### Unused function ###
def get_frames(video_path: Union[str, Path]):
    """
    Returns the list of frames from a video path
    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        print("Appending frame")
        frames.append(frame)

    return frames


def display_video_from_frames(frames: List, window_name: str='frame'):
    for frame in frames:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(25)

    cv2.destroyAllWindows()