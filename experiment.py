#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:59:23 2023

@author: tanzira, saimon, sabastain
"""
#%%All imports

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import sys
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using {device=}')

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def get_model(model_name,
              entrypoint = 'pytorch/vision:v0.10.0', 
              pretrained = True,
              weights = 'IMAGENET1K_V1'):
    if pretrained:
        model = torch.hub.load(entrypoint, model_name,
                               weights = weights)
    else:
        model = torch.hub.load(entrypoint, model_name, pretrained = False)
    return model


def preprocess_labels():
    
    data = pd.read_csv('data/archive/ISIC_2019_Training_GroundTruth.csv')
    data = data.drop('UNK', axis = 1) #no samples in this class
    data = data[~data.image.str.contains('downsampled')] #removing downsampled images

    data['labels'] = data.iloc[:, 1:-1].idxmax(axis = 1)
    classes_to_int = {v:i for i, v in enumerate(data.columns[:-1])}
    # int_to_classes = {i:v for i, v in enumerate(data.columns[:-1])}
    data['labels'] = data['labels'].map(classes_to_int)
    # num_classes = len(classes_to_int)
    print(data.head())
    #%Divide the data into train test and validation set
    x_train, x_test = train_test_split(data, test_size=0.2, stratify=data['labels'], random_state=42)
    # x_valid, x_test = train_test_split(x_test, test_size=0.2, stratify=x_test['labels'], random_state=42)

    x_train.reset_index(drop=True, inplace=True)
    # x_valid.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    return x_train, x_test


#% Extend Dataset class from torchvision and process training and test files
class CustomImageDataSet(Dataset):
    
    def __init__(self, labels, image_dir, transforms):
        self.labels = labels
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, ix):
        row = self.labels.loc[ix].squeeze()
        image = Image.open(self.image_dir + row['image'] + '.jpg')
        image = self.transforms(image)#['image']
        label = torch.as_tensor(row['labels'], dtype = torch.int64)
        return image, label


def load_dataset(datasetName, batch_size, transform):
    
    datsetNames = {'CIFAR10' : torchvision.datasets.CIFAR10,
                   'CIFAR100' : torchvision.datasets.CIFAR100,
                   'ImageNet' : torchvision.datasets.ImageNet,
                   'SVHN' : torchvision.datasets.SVHN,
                   'ICSC' : None
                   }
    dataset = datsetNames[datasetName]
    if datasetName == 'ImageNet':
        #trainset = dataset(root = './data', split = 'train', transform = transform)
        # Don't load train set
        trainset = None
        testset = dataset(root = './data', split = 'val', transform=transform)
        trainloader = None
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        return trainloader, testloader
    elif datasetName in ['CIFAR10', 'CIFAR100']:
        dataset = datsetNames[datasetName]
        trainset = dataset(root = './data', train = True, download = True, transform = transform)
        testset = dataset(root = './data', train = False, download = True, transform=transform)
    elif datasetName == 'SVHN':
        trainset = dataset(root = './data', split = 'train', download = True, transform = transform)
        testset = dataset(root = './data', split = 'test', download = True, transform = transform)
        
    elif datasetName == 'ICSC':
        image_directory = 'data/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input/'
        x_train_label, x_test_label = preprocess_labels()
        trainset = CustomImageDataSet(x_train_label, image_directory, transform)
        # valid_ds = ISICDataset(x_valid, transform)
        testset = CustomImageDataSet(x_test_label, image_directory, transform)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def train_epoch(trainloader, model, lossfunction, optimizer):
    model.train()
    size = len(trainloader.dataset)
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfunction(outputs, labels)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        num_data = (i + 1) * len(inputs)
        if i % 1000 == 0:
            print(f"current_loss: {current_loss:>7f}  [{num_data:>5d}/{size:>5d}]")
            
def train_model(model, epochs, trainloader, testloader, optimizer, scheduler, lossfunction):
    global_start_time = time.time()
    for epoch in range(epochs): 
        start_time = time.time()
        train_epoch(trainloader, model, lossfunction, optimizer)
        scheduler.step()
        end_time = time.time()
        print(f'Epoch {epoch}, LRate={scheduler.get_last_lr()[0]}, Time: {format_time(end_time - start_time)}')
        if epoch % 10 == 0:
            print(test_model(testloader, model, lossfunction))
    print("Total training time: ", format_time(end_time - global_start_time))

def test_model(testloader, model, lossfunction):
    size = len(testloader.dataset)
    correct = 0
    total = 0
    test_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            loss = lossfunction(outputs, labels)
            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            end_time = time.time()
    report_str = '\n'.join([
    f'Test images: {size}, mean loss: {test_loss / size: .2f}, accuracy: {100 * correct / total: .2f} %',
    "Total test time: " + format_time(end_time - start_time)])
    return report_str

def get_experiment_config(datasetName, modelName, test_run, pretrained = True):
    if 'efficientnet' in modelName.lower():
        entrypoint = 'NVIDIA/DeepLearningExamples:torchhub'
        model = get_model(modelName, entrypoint, pretrained = pretrained)
    else:
        model = get_model(modelName, pretrained = pretrained)
    # Replicate the hyperparameters of the DenseNet paper
    if datasetName in ['CIFAR10', 'CIFAR100']:
        epochs = 300
        learning_rate = 0.1
        batch_size = 64
        milestones = [int(epochs * 0.50), int(epochs * 0.75)]
        if datasetName == 'CIFAR10':
            model.classifier = nn.Linear(model.classifier.in_features, 10, bias = True)
        elif datasetName == 'CIFAR100':
            model.classifier = nn.Linear(model.classifier.in_features, 100, bias = True)
    elif datasetName == 'SVHN':
        epochs = 40
        learning_rate = 0.1
        batch_size = 64
        milestones = [int(epochs * 0.50), int(epochs * 0.75)]
        model.classifier = nn.Linear(model.classifier.in_features, 10, bias = True)
    elif datasetName == 'ImageNet':
        learning_rate = 0.1
        if modelName == 'densenet161':
            epochs = 90
            milestones = [30, 60]
            batch_size = 128
        else:
            epochs = 90
            milestones = [90]
            batch_size = 256
    else:
        print('Using default parameters')
        epochs = 40
        learning_rate = 0.1
        batch_size = 64
        milestones = [int(epochs * 0.50), int(epochs * 0.75)]
        
    if torch.cuda.is_available():
        model.cuda()
    transform = transforms.Compose(
        [
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
         ])
    optimizer = optim.SGD(model.parameters(), 
                          lr = learning_rate,
                          dampening = 0,
                          weight_decay= 10**(-4), 
                          nesterov = True, 
                          momentum = 0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones)  
    lossfunction = nn.CrossEntropyLoss()
    if test_run:
        epochs = 1
        
    report_str = '\n'.join([
     '-----------------------------------',
     'Experiment Configuration',
     '-----------------------------------',
    f'Model:                 {modelName}',
    f'Parameters:            {sum(p.numel() for p in model.parameters())}',
    f'Dataset:               {datasetName}',
     'Training epochs:' + ('[TEST] ' if test_run else '       ') + str(epochs),
    f'Initial learning rate: {learning_rate}',
    f'Mini-batch size:       {batch_size}',
    f'LRrate update epochs:  {milestones}', 
    '-----------------------------------'])
    print(report_str)
    return model, epochs, learning_rate, batch_size, transform, optimizer, scheduler, lossfunction, report_str


#%% Main loop

try:
    # For command line executing using bash scripts on Arc nodes
    modelName, datasetName = sys.argv[1], sys.argv[2]
    print('New node with {}, {}'.format(modelName, datasetName))
except:
    print('No command line arguments specified')
    modelName = 'densenet121' # densenetDDD | resnetDD | efficientnet
    datasetName = 'CIFAR10' # CIFAR10 | CIFAR100 | SVHN | ImageNet

test_run = False
train = True
save_model = False

model, epochs, learning_rate, batch_size, \
transform, optimizer, scheduler, lossfunction, report_str \
    = get_experiment_config(datasetName, modelName, test_run)
trainloader, testloader = load_dataset(datasetName, batch_size, transform)
if train:
    train_model(model, epochs, trainloader, testloader, optimizer, scheduler, lossfunction)

test_report_str = test_model(testloader, model, lossfunction)
test_report_str = '\n'.join(['-----------------------------------',
                             'Test Accuracy at Completion',
                             '-----------------------------------',
                             test_report_str,
                             '-----------------------------------'])
print(test_report_str)

with open('performance/{}__{}__{}_epochs.txt'.format(modelName, datasetName, epochs), 'w') as f:
    f.write(report_str)
    f.write('\n')
    f.write('Test Accuracy\n')
    f.write(test_report_str)
    f.close()
path = 'saved_models/{}__{}__{}_epochs.pt'.format(modelName, datasetName, epochs)
if (not test_run) and save_model:
    torch.save(model.state_dict(), path)

