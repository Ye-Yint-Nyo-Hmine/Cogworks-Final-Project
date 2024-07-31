import torch
import torchvision
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Union
from PIL import Image
import random
from model import *
import pickle

### FasterRCNN_MobileNet_V3_Large_320_FPN model, weights, classes, and transforms ###
model, weights, classes, transforms = get_model_weights_classes_transforms()


def detect_video(video_path: Union[str, Path], min_score: float=0.6):
    """
    Parameters:
        video_path: Union[str, Path]
            Full file path of desired video to be detected

        min_score: float, Optional
            Minimum score needed for an 'detected' object to have a bounding box drawn around it
    """
        
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = transforms(Image.fromarray(input)).unsqueeze(0) # add batch dimension

        model.eval()
        with torch.inference_mode():
            output = model(input)[0] # get first output because we only passed one frame/image in

        boxes, labels, scores = filter_boxes_labels_scores(output["boxes"], output["labels"], output["scores"], min_score=min_score)
        boxes_drawn = draw_boxes_on_frame(frame, boxes, labels, scores, classes)

        cv2.imshow(os.path.basename(video_path), boxes_drawn)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_camera(min_score: float=0.6):
    """
    Parameter:
        min_score: float, Optional
            Minimum score needed for an 'detected' object to have a bounding box drawn around it
    """

    video_capture = cv2.VideoCapture(0)
    #raw_capture = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 640, 360
    video_capture.set(3, WIDTH)
    video_capture.set(4, HEIGHT)

    while True:
        # capture the video
        ret, frame = video_capture.read()

        if ret:
            input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input = transforms(Image.fromarray(input)).unsqueeze(0) # add batch dimension

            model.eval()
            with torch.inference_mode():
                output = model(input)[0] # get first output because we only passed one frame/image in

            boxes, labels, scores = filter_boxes_labels_scores(output["boxes"], output["labels"], output["scores"], min_score=min_score)
            boxes_drawn = draw_boxes_on_frame(frame, boxes, labels, scores, classes)

            cv2.imshow("Camera Detection", boxes_drawn)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()