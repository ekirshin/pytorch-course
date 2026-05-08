#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:17:31 2026

@author: ek
"""
"""
Programming Assignment: Fruit Quality Inspection and Generation
In the world of automated food inspection, trusting your model is just as important as its accuracy. Before a system can responsibly sort fresh produce from spoiled, you need to understand why it makes those decisions. Does your model categorize a fruit as "Rotten" because of a visible mold spot, or is it cheating by looking at the background color?

This assignment challenges you to dig inside the brain of a deep learning model. You will work with a pre-trained computer vision system designed to inspect fruit. Your goal is to build a suite of visualization tools to examine its internal features, explain its decisions, and even generate new training data when real samples are scarce.

This is a journey from the inside out. You will start by dissecting the internal feature maps of the model to see how it builds understanding from pixels. Then, you will move to decision interpretability to pinpoint exactly which parts of an image drive its predictions. Finally, in the optional section, you will step beyond analysis into creation, using generative AI to synthesize realistic fruit images from scratch.

Specifically, you will perform the following steps:

Visualizing Feature Hierarchy: You will peel back the layers of a Convolutional Neural Network (CNN) to observe how it transforms raw pixel data into abstract features like edges, textures, and shapes.
Pinpointing Sensitivity with Saliency: You will implement Saliency Maps to calculate the gradient of the prediction of the model with respect to the input pixels, identifying exactly which tiny details triggered the decision.
Mapping Attention with Grad-CAM: You will use Class Activation Mapping to generate heatmaps that highlight the specific regions of the fruit that drove the classification.
Generating Synthetic Data: Finally, in an optional section, you will leverage Stable Diffusion to solve the problem of limited training data by generating realistic images of rare food defects.
Let's begin building the tools to see what your AI sees!
"""
import torch
from torch.nn import functional as F

import gc
from pathlib import Path

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import helper_utils
# import unittests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# Root path for healthy and rotten apple, mango, and tomato images
dataset_path = "./fruits_subset/"

# %%
# Visualize random healthy and rotten samples from the dataset
helper_utils.plot_samples_from_dataset(dataset_path)


# %%# Load the pre-trained weights for the ResNet-50 model
fruits_model = helper_utils.load_model("./models/fruits_quality_model.pth", device)
# Move the model to device
fruits_model = fruits_model.to(device)

# %%
# =============================================================================
# It is often helpful to inspect the architecture you are working with. The display_model_architecture function below prints a summary of the ResNet-50 layers, showing each layer's name, type, and number of parameters.
# 
# Pay attention to how the network is structured:
# 
# Early Layers: Convolutional layers that capture small, low-level details.
# Deeper Layers: Sequential blocks that combine features into larger, abstract patterns.
# Understanding this map is essential for the next steps. To visualize activations or generate heatmaps, you need to know exactly where to look. By identifying specific layers now, you will be able to choose the most informative points to monitor later when interpreting the model's decisions.
# 
# Inspect the model architecture to understand the layers you will be probing.
# 
# =============================================================================
# Display the model architecture
helper_utils.display_model_architecture(fruits_model, save_path="model.html")

# %% Making a Prediction
# =============================================================================
# Now, let's put the inspector to work. You will use an interactive tool to feed images from your dataset into the model and observe the results in real time.
# 
# Run the cell below to launch the prediction interface.
# Select a directory from the dropdown menu (e.g., Apple_Rotten) to view the predictions for that specific fruit category.
# You can switch between different directories to audit different fruits without needing to re-execute the cell.
# =============================================================================
# Launch the prediction interface
helper_utils.predict_fruit_quality(fruits_model, dataset_path, device)

# %% 2 - Visualizing Internal Representations
# =============================================================================
# You have seen the final decision of the model, but how did it arrive there?
# 
# A Convolutional Neural Network (CNN) does not see an "apple" or a "tomato" all at once. It builds this understanding hierarchically. The early layers typically act as edge and texture detectors, identifying simple curves or color gradients on the skin of the fruit. As the data flows deeper, these simple features are combined into more complex patterns, eventually recognizing shapes like a stem, a bruise, or a patch of mold.
# 
# In this section, you will peel back the layers of your network to observe this transformation in real time.
# 
# 
# 2.1 - Hooking into the Hierarchy
# To visualize these internal states, you need a way to access the data flowing through the network. By default, PyTorch discards intermediate feature maps (the outputs of hidden layers) to save memory once the forward pass is complete. To peek inside, you need a hook.
# 
# A hook is a function that you register to a specific layer. It acts like a wiretap: every time that layer processes an input, the hook intercepts the output and saves a copy of it before the data moves on.
# 
# You will use the grab helper function to create these hooks. When called with the activations dictionary and a name, it returns a closure that, once attached to a layer, saves that layer's output tensor into activations under the specified name. The function explicitly uses .detach() to remove the tensor from the computation graph, ensuring it does not track gradients, which saves memory and makes the data easier to handle.
# 
# In short, this tool allows you to capture and label feature maps from chosen layers, making it possible to inspect how the network processes an image at different stages.
# 
# Run the cell below to define the grab helper function.
# =============================================================================
def grab(activations, name):
    """
    Creates a forward hook function to capture and store the output of a specific layer.

    Arguments:
        activations: A dictionary where the captured layer output will be stored.
        name: The key under which the output tensor will be saved in the dictionary.

    Returns:
        _hook: The closure function to be registered as a hook.
    """
    # Define the internal hook function following the PyTorch hook signature (module, input, output)
    def _hook(_, __, out): 
        # Detach the output tensor from the gradient graph and store it in the dictionary
        activations[name] = out.detach()
        
    # Return the closure to be registered as a hook
    return _hook

# %% 2.2 - Capturing the Hierarchy
# =============================================================================
# Your first objective is to build a tool that captures the state of the network at various depths. This will allow you to take a single image of a fruit and extract a "fingerprint" of how that image is represented at the beginning, middle, and end of the network.
# 
# 
# Exercise 1 - cnn_feature_hierarchy
# Implement cnn_feature_hierarchy to visualize how the internal representations of the network evolve. This function uses a pre-trained model (with a ResNet-50 backbone) and a pre-processed input tensor to capture feature maps from various depths of the network.
# 
# Assume Preprocessed Input
# 
# img is already:
# RGB
# center cropped to 224×224
# normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# shaped (1, 3, 224, 224) and placed on the same device as model.
# Your Task:
# 
# Identify the specific layers you want to inspect. You need to target the following five points in the model:
# conv1
# The first convolution of the first bottleneck block in:
# layer1 (layer1[0].conv1)
# layer2 (layer2[0].conv1)
# layer3 (layer3[0].conv1)
# layer4 (layer4[0].conv1)
# Register a forward hook on each of these layers using the grab function.
# Store the handle returned by register_forward_hook in a list so you can access it later.
# =============================================================================

# GRADED FUNCTION: cnn_feature_hierarchy

def cnn_feature_hierarchy(img, model):
    """
    Visualizes the feature hierarchy of a CNN by capturing feature maps 
    from specific layers during a forward pass.

    This function attaches hooks to key convolutional layers in a ResNet-style 
    architecture to extract intermediate representations.

    Arguments:
        img: The input tensor (image) to process.
        model: The pretrained neural network module to use for feature extraction.

    Returns:
        activations: A dictionary mapping layer names to their captured 
                     feature-map tensors.
    """

    # Initialize an empty dictionary to store the captured activations
    activations = {}
    
    ### START CODE HERE ###

    # Define a dictionary mapping descriptive names to the specific model layers to probe
    layers = { 
        # Register forward hook for the first convolution layer
        "conv1": model.conv1,
        # Register forward hook for the first convolution layer in the first layer of the model
        "layer1": model.layer1[0].conv1,
         # Register forward hook for the first convolution layer in the second layer of the model
        "layer2": model.layer2[0].conv1,
        # Register forward hook for the first convolution layer in the third layer of the model
        "layer3": model.layer3[0].conv1,
        # Register forward hook for the first convolution layer in the fourth layer of the model
        "layer4": model.layer4[0].conv1
    } 

    ### END CODE HERE ###

    # Initialize a list to track the registered hook handles for cleanup
    hooks = []
    
    ### START CODE HERE ###

    # Iterate through the dictionary to register hooks on each selected layer
    for name, layer in layers.items():
        
        # Create a specific hook closure for the current layer using the 
        # `grab` helper function
        hook_function = grab(activations, name)
        
        # Register the forward hook on the layer and store the returned handle
        hook_handle = layer.register_forward_hook(hook_function)
        
        # Append the `hook_handle` to `hooks` list
        hooks.append(hook_handle)
    
    ### END CODE HERE ###

    # Perform the forward pass to trigger the hooks and capture data
    with torch.no_grad():
        _ = model(img) 

    # Remove all hooks to clean up the model and prevent memory leaks
    for h in hooks:  
        h.remove() 

    return activations

# %%
# Verify your implementation

# Load and preprocess a sample image
image_path = "./fruits_subset/Apple_Healthy/FreshApple_3.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Compute the activations
activations = cnn_feature_hierarchy(
    img=img,
    model=fruits_model
)

# Check all keys and shapes
print("Activations Keys and Shapes:\n")
for name, tensor in activations.items():
    print(f"{name}:\t{tensor.shape}")
