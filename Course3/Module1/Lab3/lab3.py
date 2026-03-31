#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 L3
Created on Mon Mar 30 20:38:17 2026

@author: ek
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchinfo

import helper_utils

# Set seed
SEED = 42

# %%
# Setup device priority: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

# %%
# Define the path to the root directory where the image dataset is stored.
dataset_path = "./UCMerced_LandUse/Images/"

# Create an easy to read list of class names for use in plots and labels.
class_names = ['Agricultural', 'Baseball Diamond', 'Buildings', 'Dense Residential',
               'Harbor', 'Medium Residential', 'Mobile Home Park', 'Parking Lot',
               'Runway', 'Sparse Residential', 'Storage Tanks', 'Tennis Court', 
               'Airplane', 'Beach', 'Chaparral', 'Forest', 'Freeway', 'Golf Course',
               'Intersection', 'Overpass', 'River'
              ]

# %% Assemble Your Data Pipeline
# Define your pipelines of transformations for training and validation data.
# Use the pre-calculated mean and std of this dataset.
# Pre-calculated mean and std of this dataset
mean = [0.485, 0.490, 0.451]
std = [0.214, 0.197, 0.191]

# Transformations for the training set (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Transformations for validation set (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])

# %% # Create the training and validation datasets by splitting the main dataset.
train_dataset, val_dataset = helper_utils.create_datasets(
    dataset_path, 
    train_transform,
    val_transform,
    train_split=0.8,
    seed=SEED
)

# Determine the number of unique classes from the dataset's properties.
num_classes = len(train_dataset.classes)

# Print a summary of the dataset split.
print(f"Total Number of Classes:  {num_classes}")     
print(f"Training set size:        {len(train_dataset)}")
print(f"Validation set size:      {len(val_dataset)}")

# %% # Define the number of images to process in each batch.
batch_size = 32

# Create the training and validation DataLoaders using the helper function.
train_loader, val_loader = helper_utils.create_dataloaders(train_dataset, val_dataset, batch_size)

# %% Visualize Training Samples
# Display the sample images from train set
helper_utils.show_sample_images(train_dataset, class_names)

# %% DenseLayer: The Engine of Feature Reuse

