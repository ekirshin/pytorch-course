# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:02:56 2026

C3 M4 A1. Programming Assignment: CleanVision AI - Optimizing Models for Metro City's Smart Fleet

Your Mission: Smart City Cleanup Initiative 🏙️

Welcome to CleanVision AI!

Congratulations on joining CleanVision AI as a Machine Learning Engineer! Our startup has just landed a major contract with the Metro City Council to revolutionize urban waste management. Here's the situation:

The Challenge:
Metro City wants to deploy AI-powered cameras on their existing fleet of 500 street cleaning vehicles to automatically identify areas that need attention. The city has provided us with a pre-trained ResNet-18 model that accurately classifies street scenes into three categories: clean, litter, and recycle. It works perfectly in the cloud—but there's a catch.

The Problem:
The street cleaning vehicles are equipped with edge devices (similar to Raspberry Pi) with:

    ❌ No GPU
    ❌ Limited RAM (4GB)
    ❌ Constrained storage (16GB)
    ❌ CPU-only inference capabilities

The city's budget won't allow for hardware upgrades, and they need real-time inference (< 50ms per image) to make instant routing decisions. Our current model is 512 MB and takes over 900ms per batch on CPU—completely unusable for their needs.

Your Task:
The city is expecting a demo in two weeks. Your manager has assigned you to optimize the model for deployment on these edge devices. The model must:

    ✅ Run efficiently on CPU-only hardware
    ✅ Fit within the storage constraints (target: < 150 MB)
    ✅ Maintain accuracy above 95% (city requirement)
    ✅ Achieve inference time under 50ms per image on CPU

The Toolkit:
Your team lead suggests a three-stage optimization pipeline:

    Stage 1: Pruning - Remove redundant weights to reduce model size
    Stage 2: Dynamic Quantization - Convert heavy layers to INT8 for faster CPU inference
    Stage 3: Quantization-Aware Training (QAT) - Fine-tune with quantization simulation for maximum accuracy retention

Success Criteria:
If you succeed, Metro City will expand the contract to 50 other cities nationwide, making CleanVision AI a leader in urban tech solutions. The CFO is counting on you—let's make this model deployment-ready!

@author: ekirshin
"""
# %%
# =============================================================================
# Introduction
# 
# You have already trained many models; now it is time to shape one for life outside a notebook. This lab guides you through taking a pre-trained ResNet-18 based StreetClassifier and turning it into a version that is easier to store, faster on CPU, and ready for deployment on edge targets. Along the way you will practice saving and reloading model state, pruning parameters, applying quantization, and evaluating the trade offs among accuracy, latency, and size.
# 
# In this lab you will:
# 
#     Load the CleanStreetDataset, initialize a pre-trained StreetClassifier, and establish a baseline evaluation.
# 
#     Work with checkpoints by saving and restoring state_dict objects for both training and inference.
# 
#     Implement magnitude-based pruning across convolutional and linear layers, with options for unstructured and structured strategies, and verify sparsity and accuracy.
# 
#     Apply dynamic quantization to linear layers for a fast CPU speedup and benchmark the effect.
# 
#     Fuse common layer patterns and prepare a quantization-aware variant, fine-tune briefly, then convert to an INT8 model.
# 
#     Compare accuracy, inference time, and file size before and after compression to understand the impact of each step.
# 
# By the end, you will have a compact classifier that keeps performance close to the original while being far more efficient to run and ship—ready for deployment on Metro City's fleet!
# 
# =============================================================================
# %% 1 - Setup and Imports
import copy
import torch
from torch.nn.utils import prune
import torch.nn as nn
import torch.ao.quantization as aoq

import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import helper_utils
#import unittests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# %% 2 - Baseline Model and Dataset
# =============================================================================
# StreetClassifier is a deep learning model designed to classify urban scene images into three categories:
# 
#     clean
#     litter
#     recycle
# 
# 2.1 Dataset
# 
# For this task, we will use the CleanStreetDataset, which is already divided into training, development, and test splits.
# In the code below, you'll see how the datasets are loaded and how data preprocessing and augmentation transforms are applied to prepare the data for training and evaluation.
# 
# =============================================================================
dataset_path = "./data/"

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
dev_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'dev'), transform=eval_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=eval_transform)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(dev_dataset))
print("Number of test samples:", len(test_dataset))
print("\nClass mapping:", train_dataset.class_to_idx)

# %% You can visualize some examples with the following helper function.
helper_utils.display_some_images(test_dataset)
