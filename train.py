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
# PURPOSE: trains the classifier for flower image using the alexnet model 
# prints out training loss, data_validation loss and data_validation accuracy
#
# From commands in the terminal: python train.py flowers/ --arc alexnet epo 8
# 
#           train.py
#All library imports:
#import matplotlib.pyplot as mplt
#import seaborn as sb
#import numpy as np
#import pandas as pd
import torch
from torch import nn
from torch import optim
from torchvision import datasets as dts
from torchvision import models as M
from torchvision import transforms as Trans
#import torch.nn.functional as F
import torch.utils.data
from collections import OrderedDict
#from PIL import Image
import argparse
import json

# creating arguments that will be used by the script
pars = argparse.ArgumentParser(description = 'Used to prepared the script')
pars.add_argument('data_dir',
                    type = str,
                    help = 'Gives directory where data is stored. Mandatory')

pars.add_argument('--save_dir',
                    type = str,
                    help = 'Provide directory to save our checkpoint. Optional')
                    
pars.add_argument('--GPU',
                    type = str,
                    default = 'cuda',
                    help = "Use GPU or not")                    

pars.add_argument('--arc',
                    type = str,
                    default = 'alexnet',
                    help = 'Use alexnet or VGG-13 as CNN. Default: alexnet')

pars.add_argument('--learn_rate',
                    type = float,
                    default = 0.001,
                    help = 'train model learn rate. default: 0.001')

pars.add_argument('--hid_units',
                    type = int,
                    default = 2048,
                    help = 'Secret unit in classifier. Default: 2048')

pars.add_argument('--epo',
                    type = int,
                    default = 8,
                    help = 'The number of default epochs 8')



#setting values for information loading
args = pars.parse_args()


data_dir = args.data_dir
training_dir = data_dir + '/train'
validation_dir = data_dir + '/valid'
testing_dir = data_dir + '/test'

# Choosing between GPU or CPU depending on user choice
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# Loading_data:
if data_dir:   # Program will load model only if mandatory argument data directory is specified
    # TODO: declare your transforms for the different process of training, data_validation and testing
    data_transforms_training = Trans.Compose([Trans.RandomRotation(30),
                                      Trans.RandomResizedCrop(224),
                                      Trans.RandomHorizontalFlip(),
                                      Trans.ToTensor(),
                                      Trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_transforms_validation = Trans.Compose([ Trans.Resize(255),
                                      Trans.CenterCrop(224),
                                      Trans.ToTensor(),
                                      Trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_transforms_testing = Trans.Compose([ Trans.Resize(255),
                                      Trans.CenterCrop(224),
                                      Trans.ToTensor(),
                                      Trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # TODO: Put each processed data in their specific folder
    image_datasets_training = dts.ImageFolder(training_dir, transform = data_transforms_training)
    image_datasets_validation = dts.ImageFolder(validation_dir, transform = data_transforms_validation)
    image_datasets_testing = dts.ImageFolder(testing_dir, transform = data_transforms_testing)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_training = torch.utils.data.DataLoader(image_datasets_training, batch_size = 64, shuffle = True)
    dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, batch_size = 64, shuffle = True)
    dataloaders_testing = torch.utils.data.DataLoader(image_datasets_testing, batch_size = 64, shuffle = True)

# mapping the integer encoded categories to the actual names of the flowers.
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# number of categories
len(cat_to_name)

#   Select the CNN based on user preference
def load_training_model(archN, hidden_units):
    if archN == 'vgg13':
        modelnet = M.vgg13(pretrained = True)
        for param in modelnet.parameters():
            param.requires_grad = False
        #If hidden_units provided:
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        #if Hiddien_units not specified:
        else:
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    #If archN NOT specified as 'vgg13', use default Alexnet
    else:
        archN = 'alexnet'
        modelnet = M.alexnet(pretrained = True)
        for param in modelnet.parameters():
            param.requires_grad = False
        # If Hidden_units specified:
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        # IF not given:
        else:
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))

    modelnet.classifier = classifier
    return modelnet, archN


#Defining validtion() function that will be used while training:
def data_validation(model, dataloaders_validation, criterion):
    model.to(device)

    validation_loss = 0
    accuracy = 0

    for ins, labs in dataloaders_validation:
        ins, labs = ins.to(device), labs.to(device)
        output = model.forward(ins)
        validation_loss += criterion(output, labs).item()

        ps = torch.exp(output)
        equality = (labs.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return validation_loss, accuracy

# Actually load model using load_training_model() function:
model, arch = load_training_model(args.arc, args.hid_units)

# Actual Training of Model:
#Initializing criterion and Optimizer:
criterion = nn.NLLLoss()
#Now for setting up Optimizer, need to check if Learning Rate has been already specified or not:
if args.learn_rate:  #if Learning Rate provided
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learn_rate)
else:         #if NOT provided, use specify default
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)


model.to(device)
# Now setting up number of Epochs to be run. Check if already specified or NOT:
if args.epo:     #if epochs give, use the same
    epochs = args.epo
else:               #if not given, use default value = 7
    epochs = 7

print_every = 40
steps = 0

for e in range(epochs):
    running_loss = 0
    for ii, (ins, labs) in enumerate(dataloaders_training):
        steps += 1
        ins, labs = ins.to(device), labs.to(device)
        optimizer.zero_grad () #where optimizer is working on classifier paramters only

        # Forward and backward passes
        outputs = model.forward (ins) #calculating output
        loss = criterion (outputs, labs) #calculating loss
        loss.backward ()
        optimizer.step () #performs single optimization step

        running_loss += loss.item () # loss.item () returns scalar value of Loss function

        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off

            # Turn off gradients for data_validation, saves memory and computations
            with torch.no_grad():
                validation_loss, accuracy = data_validation(model, dataloaders_validation, criterion)

            print("{} of {} Epochs.. ".format(e+1, epochs),
                  "Training Loss is: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss is: {:.3f}.. ".format(validation_loss/len(dataloaders_validation)),
                  "Valid Accuracy is: {:.3f}%".format(accuracy/len(dataloaders_validation)*100))

            running_loss = 0

            # Make sure training is back on
            model.train()

#Now that model has been created, save it
#switching back to normal CPU mode, as no need of GPU to save model
model.to('cpu')

#Check mapping between class name & predicted class before savig
model.class_to_idx = image_datasets_training.class_to_idx

# Create dictionary to be saved with all info:
checkpoint = {'classifier': model.classifier,
               'state_dict': model.state_dict(),
               'arch' : arch,
               'mapping' : model.class_to_idx  }

# Save using dictionary created above:
# Check if optional argument for saving directory given
if args.save_dir:  # If saving directory given, use it
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:              # If not given, use default
    torch.save (checkpoint, 'checkpoint.pth')
