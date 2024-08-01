import numpy as np
from pathlib import Path
import json
import pickle
from cogworks_data.language import get_data_path
from typing import List, Union, Sequence
from collections import Counter
from pathlib import Path
import os
from pathlib import Path
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.io import read_image


from gensim.models import KeyedVectors
from operator import itemgetter

def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    accuracy : float
    """
    predicted_labels = np.argmax(predictions, axis = 1)
    prediction_vs_truth = np.array(predicted_labels == truth)    
    fraction_correct = np.mean(prediction_vs_truth)
    return fraction_correct


class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms(antialias=True)

    def __call__(self, x):
        print(x.shape)
        y_preds = []
        for i in x:
            print(i.shape)
            print(torch.unsqueeze(i, 0).shape)
            y_pred = self.resnet18(torch.unsqueeze(i, 0))
            y_preds += y_pred

        y_preds = torch.stack([torch.tensor(j) for j in y_preds])
        
        return y_preds
        
def process_xdata(x_data, transforms):

    res = []

    for x in x_data:
        x = transforms(read_image(x))
        
        res += [x]

    return res
    

def train_model(x_data, true_labels):
    device = torch.device("cpu")
    model = ResNetModel()

    for param in model.resnet18.parameters():
        param.requires_grad = False

    num_ftrs = model.resnet18.fc.in_features
    model.resnet18.fc = nn.Linear(num_ftrs,2)

    model.resnet18 = model.resnet18.to(device)

    criterion = nn.CrossEntropyLoss()

    optimize = optim.SGD(model.resnet18.fc.parameters(), lr = 0.01, momentum = 0.9)

    x_data = process_xdata(x_data, model.transforms)

    print(len(x_data), x_data[0].shape)

    true_labels = torch.stack([torch.tensor(i) for i in true_labels])
    print(type(true_labels), true_labels)


    batch_size = 2 # idk
    acc = 0
    for epoch_cnt in range(30): # revise epoch count
        idxs = np.arange(len(x_data)) # WHAT IS X_DATA
        np.random.shuffle(idxs)  
        
        for batch_cnt in range(0, len(x_data) // batch_size):
            batch_indices = idxs[(batch_cnt * batch_size):((batch_cnt + 1) * batch_size)]

            print("idxs", len(idxs), idxs.dtype)
            print("loop", batch_cnt, batch_size, len(x_data) // batch_size)
            print("batch indices", len(batch_indices), batch_indices.dtype, batch_indices)

            batch = torch.stack([x_data[i] for i in batch_indices])
            y_true = true_labels[batch_indices]
            
            print("batch shape", batch.shape)
            outputs = model(batch) # does this work

            print("output shape:", outputs.shape, outputs)
            print("true labels shape:", true_labels.shape)
            print("y_true", y_true.shape, y_true)

            loss = criterion(outputs, y_true.to(torch.float32))

            #loss.backward()
            optimize.step()
            print(loss.item())
            
            acc += accuracy(outputs, true_labels)

    return acc / batch_cnt


