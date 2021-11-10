#!/usr/bin/env python3
#
#Reference (1) https://github.com/felixrlopezm/Udacity-project-Image-classifier
#          (2) https://github.com/Kusainov/udacity-image-classification
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
#
# PROGRAMMER: Cesar Landry Taptue Taptue
# DATE CREATED: 25/10/2021                                 
# REVISED DATE: 
# PURPOSE: makes prediction about a flower input by the user. Uses a pretrained network to return 
# the most likely flower type
#
# # From commands in the terminal: python predict.py flowers/test/1/'image name' checkpoint.pth
#
#importing required libraries:

import numpy as np
import torch
from torchvision import models as M 
from torchvision import transforms as Trans
from PIL import Image as img
import json
import argparse


#Title: <title of program/source code>
#Author: Kusainov
#Date: 11 Nov 2018
#Availability: https://github.com/Kusainov/udacity-image-classification/blob/master/predict.py

# Create arguments that will be used on the script
pars = argparse.ArgumentParser(description = "Parser for Prediction script")

pars.add_argument('img_dir', help = 'Location of the image', type = str)

pars.add_argument('loading_dir', help = 'location of the model checkpoint', type = str)

pars.add_argument('--GPU',
                    type = str,
                    default = 'cuda',
                    help = "Use GPU or not")

pars.add_argument ('--top_k',
                    type = int,
                    default = 3,
                    help = 'Top K most likely classes. (Optional)')

pars.add_argument ('--cat_names',
                    type = str,
                    help = 'Maps Categories to names' )


# TODO: Write a function that loads a checkpoint and rebuilds the model

def loading_model(file_path):
    checkpoint_pth = torch.load(file_path)
    #Checks and use the arc model 
    if checkpoint_pth['arch'] == 'alexnet':
        model = M.alexnet(pretrained = True)
    # If arc value is null, use vgg13
    else:
        model = M.vgg13(pretrained = True)
    # Load various model parameters from saved checkpoint_pth:
    model.classifier = checkpoint_pth['classifier']
    model.load_state_dict = checkpoint_pth['state_dict']
    model.class_to_idx = checkpoint_pth['mapping']

    # As model has already been tuned, switch off model tuning
    for p in model.parameters():
        p.requires_grad = False
    return model


# Emthod definition to prepare image for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = Trans.Compose([
        Trans.Resize(256),
        Trans.CenterCrop(224),
        Trans.ToTensor()])

    pil_image = img.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

# Defining predict function that will actually try and predict topkl probabilities and class of flower image as specified.
def predict(image_path, model, topkl, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)

    # Now to our feed forward model, we need to pass a tensor.
    # Thus, need to convert the numpy array to tensor
    if device == 'cuda':
        img = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(image).type(torch.FloatTensor)

    #As forward method is working with batches doing that we will have batch size = 1
    img = img.unsqueeze(dim =0)
    model.to(device)
    img.to(device)


    with torch.no_grad():
        # Pased image tensor to feedforward model
        model.eval()    #switching to evaluation mode so that dropout can function properly
        output = model.forward(img)

        # To get output as a probability
        output_prob = torch.exp(output)

        probs, indices = output_prob.topk(topkl)
        probs = probs.cpu()
        indices = indices.cpu()
        # COnvert both the above to numpy array:
        probs = probs.numpy()
        indices = indices.numpy()

        probs = probs.tolist()[0]
        indices = indices.tolist()[0]

        mapping = {val : key for key,val in model.class_to_idx.items()}

        classes = [mapping[item] for item in indices]
        classes = np.array(classes)

        return probs, classes


#Settigng values for data loading:
args = pars.parse_args()
file_path = args.img_dir

# Defining GPU or CPU:
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if args.cat_names:
    with open(args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Load model from saved checkpoint:

model = loading_model(args.loading_dir)

#Define no of classes to be predicted. Default = 3
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 3

# Calculating probabilites and classes:
probs, classes = predict(file_path, model, nm_cl, device)

class_names = [cat_to_name[item] for item in classes]

for k in range (nm_cl):
     print("k: {}/{}.. ".format(k+1, nm_cl),
            "Flower name: {}.. ".format(class_names [k]),
            "Probability: {:.3f}..% ".format(probs [k]*100),
            )
