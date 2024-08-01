import torch
import torchvision
from torchvision import transforms
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Union
from PIL import Image
import random
from torchvision.io import read_image
import main

'''
Don't want to upload the whole data set, but it's being pulled from here

https://www.kaggle.com/datasets/phylake1337/fire-dataset?resource=download

when downloaded and unzipped, parent folder should be archive
'''

data_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

def get_training_paths(training_data_size):

    all_fire_paths = os.listdir('archive/fire_dataset/fire_images')

    all_non_fire_paths = os.listdir('archive/fire_dataset/non_fire_images')
 
    indexes = np.arange(0, min(len(all_fire_paths), len(all_non_fire_paths)))

    np.random.shuffle(indexes)

    chosen_idxs = indexes[:training_data_size]

    no_fire = ['archive/fire_dataset/non_fire_images/' + all_non_fire_paths[i] for i in chosen_idxs]

    if len(chosen_idxs) < training_data_size:
        chosen_idxs += indexes[-1]

    fire = ['archive/fire_dataset/fire_images/' + all_fire_paths[j] for j in chosen_idxs]

    return fire, no_fire

    



def do_training():
    training_data_size = 5

    fire_paths, no_fire_paths = get_training_paths(training_data_size)

    '''fire_paths = ["archive/fire_dataset/fire_images/fire.1.png", 
                    "archive/fire_dataset/fire_images/fire.2.png", 
                    "archive/fire_dataset/fire_images/fire.3.png",
                    "archive/fire_dataset/fire_images/fire.4.png", 
                    "archive/fire_dataset/fire_images/fire.5.png"]

    no_fire_paths = ["archive/fire_dataset/non_fire_images/non_fire.1.png", 
                     "archive/fire_dataset/non_fire_images/non_fire.2.png", 
                     "archive/fire_dataset/non_fire_images/non_fire.3.png", 
                     "archive/fire_dataset/non_fire_images/non_fire.4.png", 
                     "archive/fire_dataset/non_fire_images/non_fire.5.png"]
    
    fire_tensors = []

    no_fire_tensors = []

    #scripted_transforms = torch.jit.script(data_transforms)

    for i in range(training_data_size):
        print(read_image(fire_paths[i]).dtype)
        fire_tensors += read_image(fire_paths[i]).to(torch.float32)
        no_fire_tensors += read_image(no_fire_paths[i]).to(torch.float32)
    
    all_training_data = torch.cat(fire_tensors + no_fire_tensors)'''

    all_training_data = fire_paths + no_fire_paths

    accuracy = main.train_model(all_training_data, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    print(accuracy)


do_training()

'''Written by Edwardia'''
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