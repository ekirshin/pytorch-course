#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 20:26:05 2026

@author: ek
"""

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import helper_utils

# Set dataset directory
root_dir = './pytorch_datasets'

#%% # Initialize the CIFAR-10 training dataset
cifar_dataset = datasets.CIFAR10(
    root=root_dir,      # Path to the directory where the data is/will be stored
    train=True,         # Specify that you want the training split of the dataset
    download=True       # Download the data if it's not found in the root directory
)

#%%
# Get the first sample (at index 0), which is a (image, label) tuple
image, label = cifar_dataset[0]

print(f"Image Type:        {type(image)}")
# Since `image` a PIL Image object, its dimensions are accessed using the .size attribute.
print(f"Image Dimensions:  {image.size}")
print(f"Label Type:        {type(label)}")

#%% 
# Define a transformations pipeline
cifar_transformation = transforms.Compose([
    transforms.ToTensor(),
    # The mean and std values are standard for the CIFAR-10 dataset
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)
                        )
])

# Assign the entire transformation pipeline to the dataset's .transform attribute
cifar_dataset.transform = cifar_transformation

#%% 
# Access the first item again
image, label = cifar_dataset[0]

print(f"Image Type:                   {type(image)}")
# Since the `image` is now a PyTorch Tensor, its dimensions are accessed using the .shape attribute.
print(f"Image Shape After Transform:  {image.shape}")
print(f"Label Type:                   {type(label)}")

#%% Use a DataLoader to grab a small batch of your transformed images and display them in a grid.
# Define the datalaoder
cifar_dataloader = data.DataLoader(cifar_dataset, batch_size=8, shuffle=True)

helper_utils.display_images(cifar_dataloader)

#%% # Define the transformation pipeline
emnist_transformation = transforms.Compose([
    # 90-degree rotation, it randomly rotates between +90 degrees and +90 degrees
    transforms.RandomRotation(degrees=(90, 90)),
    # p=1.0 guarantees vertical flip
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    # Normalizes the tensor, rescaling pixels from [0, 1] to [-1, 1]
    transforms.Normalize((0.5,), (0.5,)) # The mean and std must be in a tuple
])

#%% 
emnist_digits_dataset = datasets.EMNIST(root=root_dir,
                                        split='digits',  # Specify the 'digits' split
                                        train=False,
                                        download=True,
                                        transform=emnist_transformation
                                       )

#%% Use a DataLoader to grab a small batch of your loaded EMNIST images and display them in a grid.
emnist_digits_dataloader = data.DataLoader(emnist_digits_dataset, batch_size=8, shuffle=True)
helper_utils.display_images(emnist_digits_dataloader)

#%% Custom and Specialized Datasets
root_dir = './tiny_fruit_and_vegetable'

#%% # Define a transformation pipeline
image_transformation = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize( 
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

#%% initialize the ImageFolder dataset
fruit_dataset = datasets.ImageFolder(root=root_dir,
                                     transform=image_transformation
                                    )

#%% Use a DataLoader to grab a small batch of your loaded ImageFolder images and display them in a grid.
fruit_dataloader = data.DataLoader(fruit_dataset, batch_size=8, shuffle=True)

helper_utils.display_images(fruit_dataloader)

#%% Synthetic Fake Data
# Define a transformation pipeline
fake_data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize the FakeData dataset
fake_dataset = datasets.FakeData(
    size=1000,                    # Total number of fake images
    image_size=(3, 32, 32),       # (Channels, Height, Width)
    num_classes=10,               # Number of possible classes
    transform=fake_data_transform # Apply the transformation
)

#%% Use a DataLoader to grab a small batch of your generated FakeData images and display them in a grid.
fake_dataloader = data.DataLoader(fake_dataset, batch_size=8, shuffle=True)

helper_utils.display_images(fake_dataloader)

#%% Exercise 1: FashionMNIST
# Define a transformation pipeline
grayscale_transformation = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    # Use a mean and std for a single channel
    transforms.Normalize(mean=(0.5,), std=(0.5,)) 
])

#%% Re-define the root directory back as './pytorch_datasets'
# Set dataset directory
root_dir = './pytorch_datasets'

try:
    fashion_mnist_dataset = datasets.FashionMNIST(root=root_dir,
                                        train=False,
                                        download=True,
                                        transform=grayscale_transformation
                                       )
    
    print("\033[92mDataset loaded successfully!")
    
except:
    print("\033[91mSomething went wrong, try again!")
    
#%% To confirm that everything has loaded correctly, visualize a small batch of the images.
fashion_mnist_dataloader = data.DataLoader(fashion_mnist_dataset, batch_size=8, shuffle=True)   
 
helper_utils.display_images(fashion_mnist_dataloader)

#%% Exercise 2: SVHN
# Define a transformation pipeline.
# Define a 3-channel transformation
svhn_transformation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#%% 
try:
    svhn_dataset = datasets.SVHN(root=root_dir,
                                 split='train',
                                 download=True,
                                 transform=svhn_transformation
                                 )
    
    print("\033[92mDataset loaded successfully!")
    
except:
    print("\033[91mSomething went wrong, try again!")
    
#%% To confirm that everything has loaded correctly, visualize a small batch of the images.
svhn_dataloader = data.DataLoader(svhn_dataset, batch_size=8, shuffle=True)

helper_utils.display_images(svhn_dataloader)