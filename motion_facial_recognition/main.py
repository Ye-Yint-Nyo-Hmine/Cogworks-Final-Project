"""
*Here is where all the functions will come together, with the interface
*Face detection model is defined here too.
"""
from facenet_models import FacenetModel
from user_profile import *

import skimage.io as io # reading an image file in as a numpy array
import numpy as np
import pickle # for database
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
import sklearn
from camera import take_picture

from pathlib import Path
import os
from typing import Union, List

import tkinter as tk
from tkinter import messagebox

# this will download the pretrained weights for MTCNN and resnet
# (if they haven't already been fetched)
# which should take just a few seconds
model = FacenetModel()

# Initialize database
image_path = Path("imgs")

database = {}
with open('database.pkl', 'rb') as f:
    database = pickle.load(f)

threshold = None
with open('threshold.txt', 'r') as f:
    threshold = float(f.readline().strip())



def add_descriptor_vectors_to_database(descriptor_vectors: np.ndarray, names: List[str]):
    global database

    """descriptors_vectors is a shape (N, 512) array and names is a length N list."""

    assert len(descriptor_vectors) == len(names), f"length of descriptor_vectors and names do not match | length of vectors: {len(descriptor_vectors)}, length of names: {len(names)}"

    for name, descriptor_vector in zip(names, descriptor_vectors):
        if name in database:
            profile = database[name]
            profile.add_descriptors(descriptor_vector[np.newaxis, :])
        else:
            profile = Profile(name, descriptors=descriptor_vector[np.newaxis, :])
            database[profile.name] = profile


def image_to_rgb(image_to_rgb_path, max_width=500):
    """# shape-(Height, Width, Color)
    print(f"{image_to_rgb_path}: image path when passed to image_to_rgb")
    image = io.imread(str(image_to_rgb_path)).astype(np.uint8) # had to change it to np.float32 bc I was getting problems without it
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
        
    return image.astype(np.uint8)"""

    from PIL import Image, ImageOps
    image = Image.open(image_to_rgb_path)
    image = ImageOps.exif_transpose(image)  # Correct the image orientation based on EXIF data
    width, height = image.size
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = image.resize((new_width, new_height))

    image = np.array(image).astype(np.uint8)

    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    return image

def cos_dist(a,b):
    out = 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)) # changed from @ to * because a and b have different float types
    return out


def match(descriptor_vector: np.ndarray, threshold: float=threshold):
    
    ### descriptor_vectors.shape should be (1, 512)

    global database

    lowest_dist_and_profile = [3, None] # set to 3 because the bound is 2
    for profile in database.values():
        mean_discriptor_vector = profile.mean_descriptor # a 1-D array
        if cos_dist(descriptor_vector[0], mean_discriptor_vector) < lowest_dist_and_profile[0]:
            lowest_dist_and_profile = [cos_dist(descriptor_vector, mean_discriptor_vector), profile]
            # print(f"lowest_dist_and_profile for {profile.name}: {lowest_dist_and_profile}")

    if lowest_dist_and_profile[0] < threshold:
        if descriptor_vector not in lowest_dist_and_profile[1].descriptors:
            lowest_dist_and_profile[1].add_descriptors(descriptor_vector)
        return lowest_dist_and_profile[1].name
    else:
        return "Unknown"

def detect_faces(image, threshold=.7):
    """
    It takes in an image and return a list of
    boxes that are already filtered respect to the threshold
    """
    global model

    boxes, probabilities, landmarks = model.detect(image)
    # just assuming that this is the threshold
    
    # these are a list of boxes filtered after the threshold
    if boxes is None:
        return
    filtered = [box for box, prob in zip(boxes, probabilities) if prob > threshold] #####
    
    return filtered

def get_descriptors(image, boxes):
    return model.compute_descriptors(image, boxes)


def draw_boxes(image, boxes, names, camera: bool, object_detected:str):
    """
    It just draw boxes
    """
    if not camera: # if not or if, we'll never know
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # print(f"Image type: {type(image)}, Image shape: {image.shape}")
    for box, name in zip(boxes, names):
        start_x, start_y = int(box[0]), int(box[1])
        end_x, end_y = int(box[2]), int(box[3])
        
        color = (0, 255, 25) if name != "Unknown" else (255, 25, 0)
        image = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        image = cv2.putText(image, name, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    detection_color = (0, 25, 255)
    image = cv2.putText(image, f"Object detected: {object_detected}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 2)

    return image


def to_grayscale(image):
    ### Turn image from RGB to Grayscale ###
    (row, col) = image.shape[0:2] 
    
    # Take the average of pixel values of the BGR Channels 
    # to convert the colored image to grayscale image 
    for i in range(row): 
        for j in range(col): 
                # Find the average of the BGR pixel values 
            image[i, j] = sum(image[i, j]) * 0.33

    return image # should be 2-D array?


### Dead function
"""def main(descriptors, threshold, rgb_pic, img):
    
    # This will be the main function for the program, where it will call all the necessary functions
    # and also displays interface of some sort.

    # descriptors: np.ndarray(), shape (N, 512)
    # threshold: float, global variable for matching
    # rgb_pic: np.ndarray: shape


    boxes = detect_faces(rgb_pic)
    names = [match(descriptor[np.newaxis, :], threshold) for descriptor in descriptors]
    
    boxes_drawn = draw_boxes(img, boxes, names)

    # cv2.imshow('Some really cool and interesting name for this project', boxes_drawn)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    recognize()
"""

def main():
    video_capture = cv2.VideoCapture(0)
    #raw_capture = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 640, 360
    video_capture.set(3, WIDTH)
    video_capture.set(4, HEIGHT)
    last_mean = 0

    while True:
        # capture the video
        ret, frame = video_capture.read()

        # mirror the frame for perfect coordination
        if ret:
            frame = cv2.flip(frame, 1)
                
            frame = np.array(frame).astype(np.uint8) # shape (640, 360, 3)
            feed = frame

            ### Calculate whether there was motion ###
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = np.abs(np.mean(gray) - last_mean)
            last_mean = np.mean(gray)
            # print(result)

            if result > 0.3:
                detection_color = (0, 25, 255)
                feed = cv2.putText(feed, "Motion Detected!!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, detection_color, 2)

            boxes = detect_faces(frame)
            if boxes:
                descriptors = get_descriptors(frame, boxes) # shape (N, 512) array
                names = [match(descriptor[np.newaxis, :], threshold) for descriptor in descriptors]
                
                feed = draw_boxes(feed, boxes, names, camera=True, object_detected="Human") # image shape = (360, 640, 3)
                    

            cv2.imshow('This You?', feed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def recognize():
    global threshold

    print("SMILE!!")
    img_array = take_picture() # note: add to read me that the user will have to configure their Camera package before using this application
    print("Took a picture!")

    boxes = detect_faces(img_array)
    if boxes: #* check if boxes are empty or not
        print("Camera succesfully detected face")
        descriptor_vector = get_descriptors(img_array, boxes) # (N, 512) array

    print(f"descriptor vector of camera person(s): {descriptor_vector[:, :10], descriptor_vector.shape}")
    names = [match(descriptor[np.newaxis, :], threshold) for descriptor in descriptor_vector]
    print(f"Name(s) of person(s) in camera: {names}")

    box_drawn = draw_boxes(img_array, boxes, names)

    cv2.imshow('This You?', box_drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # vars = initialize_database()
    # descriptors, rgb_pics, imgs = vars
    # print(f"Returns of initialize_database exist? : {descriptors and rgb_pics}")
    # print(f"Length of descriptors passed to main function: {len(descriptors)} | length of images: {len(rgb_pics)}")
    # for des, rgbpic, img in zip(descriptors, rgb_pics, imgs):
    #    main(des, threshold, rgbpic, img)

    root = tk.Tk()
    root.withdraw()
    add_to_database = messagebox.askyesno("Add photos to Database?", "Do you want to take some pictures to add to our database?")

    if add_to_database:
        name = []
        def get_name():
            global name
            name = [entry.get()]
            root.destroy()

        root.deiconify()
        root.geometry("500x100+700+300")
        label = tk.Label(root, text="What's your name?")
        label.pack()
        entry = tk.Entry(root)
        entry.pack()

        button = tk.Button(root, text="Enter", command=get_name)
        button.pack()
        root.mainloop()

        root = tk.Tk()
        messagebox.showinfo("Going to take database photos", "We're going to take 5 pictures of you to add to our database. Make sure to smile!!")
        root.withdraw()

        photos = [take_picture() for i in range(5)]
        boxes = [np.array(detect_faces(photo)) for photo in photos]
        descriptor_vectors = [get_descriptors(photo, box) for photo, box in zip(photos, boxes) if box.any()]
        descriptor_vectors = np.array([dv for des_vec in descriptor_vectors for dv in des_vec]) # get shape (N, 512) array then index each shape (512,) item in the array

        name = name * len(descriptor_vectors)

        add_descriptor_vectors_to_database(descriptor_vectors, name)

    root.deiconify()
    messagebox.showinfo("Live Camera", "Enabling our live camera. To quit the program, press 'q'. Say hi to yourself!")
    root.withdraw()
    main()
