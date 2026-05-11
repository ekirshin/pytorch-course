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

# %% 
# Now that you have confirmed your code works, you can explore! The block below contains a variety of images from the dataset. Use them to see how the internal features of the model change depending on whether it looks at a fresh apple or a rotten tomato.
# Define the file path for an image for feature visualization
image_path = "./fruits_subset/Apple_Healthy/FreshApple_3.jpg"

# %%
# Preprocess the sample image
img = helper_utils.preprocess_image(image_path, device)

# Compute the activations
activations = cnn_feature_hierarchy(
    img=img,
    model=fruits_model
)

# Display the feature hierarchy for the given image
helper_utils.display_feature_hierarchy(activations, img)

# %%
# =============================================================================
# 2.3 - Processing Feature Maps
# You now have the raw data, but it is hard to interpret. Early layers have high resolution but few channels, while deeper layers have low resolution but hundreds of channels. To make sense of this, you need to standardize these representations.
# 
# In this task, you will build a compact visual summary that shows the strongest responding feature at five depths of the model. You will create a "visual strip" by automatically selecting the single most active channel, the feature the network is "shouting" about the loudest, and resizing it to match the original image. This creates a clear, side-by-side comparison of how the focus of the network shifts from simple edge and texture responses toward more abstract patterns as depth increases.
# 
# 
# Exercise 2 - feature_map_strip
# Implement feature_map_strip. This function builds a visual summary by identifying and resizing the most significant feature captured at five different depths of the network.
# 
# Your Task:
# 
# Retrieve Raw Features: Use the cnn_feature_hierarchy function you implemented in Exercise 1 to get the raw feature maps for the image.
# Process Each Layer: Iterate through the layers in order: ["conv1", "layer1", "layer2", "layer3", "layer4"]. For each layer's feature map:
# Identify the Top Channel: Calculate the mean activation across the spatial dimensions (height and width) for every channel. Find the index of the channel with the highest mean.
# Slice: (This is already implemented for you) Extract just that single channel, keeping the tensor 4D (maintaining the batch and channel dimensions).
# Upsample: Resize this single-channel feature map to 224x224 using "bilinear" interpolation. Ensure you set align_corners to False.
# Normalize: Scale the pixel values of the upsampled map to the range [0, 1] using min-max normalization. Add a small epsilon, 1e-8, to the denominator to prevent division by zero.
# Append each processed tensor to a list before returning it.
# =============================================================================
# GRADED FUNCTION: feature_map_strip

def feature_map_strip(img, model):
    """
    Processes an image through a model to extract, select, and upsample 
    representative feature maps from specific layers.

    This function retrieves raw feature activations, identifies the most 
    active channel based on mean activation, and upsamples it to match 
    the original image resolution for visualization purposes.

    Arguments:
        img: The input image tensor.
        model: The pretrained neural network module used for feature extraction.

    Returns:
        upsampled: A list of tensors, each representing the most active 
                   channel from a specific layer, resized to 224x224 and 
                   normalized to [0, 1].
    """

    ### START CODE HERE ###
    
    # Capture the raw feature maps using `cnn_feature_hierarchy` (exercise 1)
    feats = cnn_feature_hierarchy(img, model)
    
    # Initialize list to store processed maps
    upsampled = [] 

    # Iterate through the specific layers to visualize
    for name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
        
        # Extract the tensor for the current layer
        fm = feats[name]
        
        # Select the most "active" feature channel
        # Calculate the mean activation per channel (averaging over Height and Width dims)
        avg_activation = fm.mean(dim=(2,3))
        
        # Find the index of the channel with the highest average activation
        idx = avg_activation.argmax(dim=1)
        
        # Slice the tensor to keep only that specific channel (keep dims 4D: B, C, H, W)
        sel = fm[:, idx:idx+1] 
        
        # 4. Upsample the small feature map to the original image size (224x224)
        sel = F.interpolate(sel, size=(224, 224), mode="bilinear", align_corners=False)
        
        # 5. Normalize values to [0, 1] for visualization
        sel = (sel - sel.min()) / (sel.max() - sel.min() + 1e-8)
        
        # Append `sel` to list `upsampled`
        upsampled.append(sel)

    ### END CODE HERE ###

    return upsampled

# %% # Verify your implementation

# Load and preprocess a sample image
image_path = "./fruits_subset/Tomato_Rotten/rottenTomato_8.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Extract and resize the most significant activation channel from each key layer
upsampled = feature_map_strip(
    img=img,
    model=fruits_model
)

# Verify that all extracted maps have been upsampled to the input size (224x224)
print("Shape of the upsampled feature maps:\n")
print(f"conv1:  {upsampled[0].shape}")
print(f"layer1: {upsampled[1].shape}")
print(f"layer2: {upsampled[2].shape}")
print(f"layer3: {upsampled[3].shape}")
print(f"layer4: {upsampled[4].shape}")

# %%
# Now that you have confirmed your code works, you can explore! The block below allows you to run your new visualization pipeline. Use the paths below to compare how the feature hierarchy differs between a healthy fruit (smooth textures) and a rotten one (irregular patches).
# Define the file path
image_path = "./fruits_subset/Tomato_Rotten/rottenTomato_8.jpg"

# Preprocess the sample image
img = helper_utils.preprocess_image(image_path, device)

# Extract and resize the most significant activation channel from each key layer
upsampled = feature_map_strip(
    img=img,
    model=fruits_model
)

# Display a horizontal sequence of the five processed feature maps
helper_utils.visual_strip(upsampled)

# %%
# =============================================================================
# 3 - Pixel Level Scrutiny: Saliency Maps
# Visualizing feature maps showed you what patterns the network detects, but it did not tell you which of those patterns were important for the final decision. Does the model classify an apple as rotten because of the large brown spot, or because of the lighting in the background?
# 
# To answer this, you need Saliency Maps.
# 
# A saliency map visualizes the gradient of the prediction with respect to the input pixels. In simpler terms, it asks the model: "If I slightly change the color of this specific pixel, how much does your confidence in the prediction change?". This provides a detailed view of what the network is sensitive to at the input level, helping you verify that its attention is focused on the fruit rather than background clutter.
# 
# 
# Exercise 3 - saliency_map
# You will now implement saliency_map. This function generates a clear, pixel level explanation of the sensitivity of the model for a chosen class. By completing this, you will produce a heatmap where brighter values indicate exactly which pixels influence the prediction the most.
# 
# Your Task:
# 
# Prepare the Input:
# 
# Create a clone of the image_tensor and detach it from the current graph to ensure a clean history.
# Explicitly enable gradient tracking on this new tensor so PyTorch knows to track operations on it (setting the requires_grad_() flag to True).
# Forward Pass:
# 
# Run the model on this prepared tensor to get the output logits.
# Select the specific score corresponding to class_idx from the output batch.
# Backward Pass:
# 
# (The model gradients are cleared for you).
# Trigger backpropagation from the target_logit to compute the gradients.
# Compute Saliency:
# 
# Access the gradients of the input image.
# Take the absolute value of the gradients and sum them across the color channel dimension to collapse it into a 2D map.
# Normalize:
# 
# Shift the map so the minimum value is 0.
# Scale the map so the maximum value is 1 (divide by the maximum plus a small epsilon, 1e-8).
# =============================================================================
# GRADED FUNCTION: saliency_map

def saliency_map(model, image_tensor, class_idx):
    """
    Generate a saliency map for a single image and class.

    This function computes the gradients of the target class score with respect 
    to the input image pixels. The resulting map highlights which pixels usually 
    influence the model's prediction the most.

    Arguments:
        model: A trained CNN model instance; should be in evaluation mode.
        image_tensor: Input image tensor with shape (1, 3, H, W). Must be pre-processed 
                      consistently with the model's training data.
        class_idx: The integer index of the specific target class logit to explain.

    Returns:
        heatmap: A torch.Tensor 2-D saliency heat-map normalised to the 
                 range [0, 1] with shape (H, W).
    """ 

    ### START CODE HERE ###

    # Create a clone of the input tensor to avoid modifying the original data
    image_tensor = image_tensor.clone()
    # Detach the tensor from the current computation graph to start a new tracking history
    image_tensor = image_tensor.detach()
    # Enable gradient tracking for the input tensor to allow backpropagation to the pixels
    image_tensor.requires_grad_()

    # Perform a forward pass of the image through the model
    output = model(image_tensor)
    # Extract the logit (raw score) corresponding to the target class index
    target_logit = output[0, class_idx]
    
    ### END CODE HERE ###

    # Clear any existing gradients in the model parameters
    model.zero_grad()

    ### START CODE HERE ###
    
    # Perform the backward pass to compute gradients of the target logit w.r.t the input
    target_logit.backward()

    # Compute the absolute value of the gradients and sum across the color channels (C dim)
    grads = image_tensor.grad.abs().sum(dim=1)[0]

    # Normalize the gradients to the [0, 1] range for visualization
    # Subtract the minimum value to shift the range to start at 0
    grads -= grads.min()
    # Divide by the maximum value (plus a small epsilon) to scale to [0, 1]
    grads /= grads.max() + 1e-8

    ### END CODE HERE ###

    # Detach the resulting heatmap from the computation graph
    heatmap = grads.detach()

    return heatmap

# %% # Verify your implementation

# Load and preprocess a sample image
image_path = "./fruits_subset/Apple_Rotten/rottenApple_7.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Define the target category for explanation (1 corresponds to 'rotten')
class_idx = 1

# Compute saliency map
heatmap = saliency_map(
    model=fruits_model,
    image_tensor=img,
    class_idx=class_idx
)

# Confirm the heatmap matches input dimensions and is normalized to [0, 1]
print("Shape and Range of the heatmap:\n")
print(f"Shape: {heatmap.shape}")
print(f"Range: min = {heatmap.min()}, max = {heatmap.max()}")

# %%
# Now that you have confirmed your code works, you can explore! The block below allows you to run your saliency analysis on different samples. Use the paths below to see if the model focuses on the actual defects (like rot spots) or if it gets distracted by irrelevant background details.
# Define the file path
image_path = "./fruits_subset/Apple_Rotten/rottenApple_7.jpg"

# Define the target category for explanation 
# (0 corresponds to 'fresh', 1 corresponds to 'rotten')
class_idx = 1

# Preprocess the sample image
img = helper_utils.preprocess_image(image_path, device)

# Compute saliency map
heatmap = saliency_map(
    model=fruits_model,
    image_tensor=img,
    class_idx=class_idx
)

# Display saliency map
helper_utils.display_saliency(image_tensor=img, heatmap=heatmap)

# %%
# =============================================================================
# Interpreting Saliency Maps: What to Look For
# 
# When examining your results, keep these interpretation guidelines in mind:
# 
# Focus of Attention: Bright regions indicate pixels that strongly influence the prediction of the model for the target class. For a "rotten" classification, you should expect to see highlights on damaged or discolored areas of the fruit.
# Expected Behavior: A well-trained model should highlight relevant features (e.g., brown spots, mold, holes) rather than background elements. If the saliency focuses heavily on the background, the model might be relying on spurious correlations.
# Noise vs. Signal: Saliency maps can be visually noisy, you will often see scattered bright pixels. Focus on the overall pattern rather than individual points. Look for coherent clusters of sensitivity.
# Complementary to CAM: Saliency maps provide pixel-level sensitivity (fine-grained), while CAM shows region-level importance (coarse-grained). Use both together for a complete picture. Saliency tells you exactly which pixels, while CAM tells you which general areas.
# Limitations:
# Saliency maps show sensitivity, not strictly causation.
# Sharp edges often appear salient simply because they represent high-frequency changes, even if they are not semantically critical.
# =============================================================================

# %% 4 - Regional Attention: Class Activation Maps
# =============================================================================
# Saliency maps are powerful, but they can be visually noisy. They show you every single pixel that matters, which often results in a scattered "star map" of high-contrast edges. Sometimes, you want a broader answer. Instead of asking "which pixel matters?", you want to ask "which region matters?"
# 
# For this, you use Class Activation Maps (CAM).
# 
# CAMs work by combining the feature maps from the very last convolutional layer, where the model has its most advanced understanding of shapes and objects, with the final classification weights. This produces a smooth heatmap that highlights the entire object or region the model is focusing on.
# 
# Why "Simplified" CAM?
# 
# In this section, you will use a direct computation method rather than the gradient-based approach (Grad-CAM) often seen in generic tools. This is possible because your ResNet-50 inspector uses a specific architectural pattern: Global Average Pooling (GAP) followed by a fully connected layer. This structure allows you to mathematically map the weights of the final layer directly back onto the feature maps, offering a clean and efficient way to visualize attention without the need for backpropagation.
# 
# 
# Exercise 4 - simplified_cam
# You will now implement simplified_cam. This function generates a clear, region-level view of the evidence the model uses for a specific class, complementing the pixel-level sensitivity you obtained with saliency maps.
# 
# Your Task:
# 
# Register Hook:
# 
# Attach the save_fmap hook (provided for you) to the third convolution of the final bottleneck block in layer4 (model.layer4[-1].conv3).
# Compute CAM:
# 
# Retrieve Features - Extract the captured feature maps from the fmap_holder dictionary.
# Get Weights - Access the weight vector for the specific class_idx from the model's fully connected layer (model.fc.weight).
# Weighted Sum - Compute the dot product between the class weights and the feature channels. You should sum over the channel dimension to produce a single 2D map. torch.einsum is an efficient tool for this.
# ReLU - Apply ReLU to the resulting map to keep only positive contributions (implemented for you).
# Normalize - Scale the map to the range [0, 1] (min/max normalization). Use epsilon as 1e-8.
# Upsample:
# 
# Resize the map to match the input image dimensions (H, W).
# Note: F.interpolate requires a 4D input (Batch, Channel, Height, Width), but your map is currently 2D. You will need to add two dummy dimensions before interpolating.
# 
# =============================================================================
def simplified_cam(model, image_tensor, class_idx):
    """
    Generates a simplified Class Activation Map (CAM) for a specific image and class.

    This function extracts the feature maps from the final convolutional layer 
    and computes a weighted sum using the weights from the final fully connected 
    layer. The resulting map highlights regions of the image that contributed 
    most to the prediction of the target class.

    Arguments:
        model: A trained ResNet-style neural network module.
        image_tensor: The input image tensor (1, 3, H, W), normalized for the model.
        class_idx: The integer index of the target class to explain.

    Returns:
        heatmap: A 2-D tensor representing the class activation heatmap, 
                 scaled to [0, 1] with the same spatial dimensions as the input.
    """

    # Initialize an empty dictionary to store the captured feature maps
    fmap_holder = {}

    # Define a hook function to detach and store the layer output during the forward pass
    def save_fmap(_, __, output): 
        fmap_holder["feat"] = output.detach()

    ### START CODE HERE ###

    # Register the forward hook on the final convolutional layer to capture features
    hook = model.layer4[-1].conv3.register_forward_hook(save_fmap)

    ### END CODE HERE ###

    # Perform a forward pass with the image to trigger the hook
    with torch.no_grad():
        _ = model(image_tensor) 

    # Remove the hook to clean up the model and stop capturing data
    hook.remove() 

    ### START CODE HERE ###
    
    # Retrieve the captured feature maps from the dictionary
    feats = fmap_holder["feat"]
    # Extract the weight vector corresponding to the target class from the FC layer
    weight_vec = model.fc.weight[class_idx]

    # Compute the weighted sum of feature maps along the channel dimension
    # Uses Einstein summation: 'c' (channels), 'chw' (features) -> 'hw' (spatial map)
    cam = torch.einsum("c,chw->hw", weight_vec, feats.squeeze(0))

    # Apply ReLU to retain only positive contributions to the class score
    cam = F.relu(cam) 
    # Normalize the activation map values to the range [0, 1]
    cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-8)

    ### END CODE HERE ###

    # Retrieve the spatial dimensions (Height, Width) of the original input
    H, W = image_tensor.shape[2:]

    ### START CODE HERE ###

    # Upsample the low-resolution activation map to match the input image size
    cam_up = F.interpolate( 
        # Add batch and channel dimensions required for interpolation (1, 1, H, W)
        cam.unsqueeze(0).unsqueeze(0),
        # Specify the target output size matching the input image
        size=(H, W),
        # Use bilinear interpolation for smooth resizing
        mode="bilinear", 
        # Disable corner alignment to align the geometric centers of pixels
        align_corners=False, 
    )[0, 0] 

    ### END CODE HERE ###

    # Detach the result from the graph and move to CPU if necessary
    heatmap = cam_up.cpu().detach()

    return heatmap

# %% # Verify your implementation

# Load and preprocess a sample image
image_path = "./fruits_subset/Apple_Rotten/rottenApple_5.jpg"
img = helper_utils.preprocess_image(image_path, device)

# Define the target category for explanation (1 corresponds to 'rotten')
class_idx = 1

# Compute CAM
heatmap = simplified_cam(
    model=fruits_model, 
    image_tensor=img, 
    class_idx=class_idx
)

# Verify shape and range
print("Shape and Range of the CAM:\n")
print(f"Shape: {heatmap.shape}")
print(f"Range: min = {heatmap.min()}, max = {heatmap.max()}")

# %% 
# Now that you have confirmed your code works, you can explore! The block below allows you to generate heatmaps for different fruits. Use the paths below to see if the model's "attention" aligns with the rotten spots on the fruit, or if it is looking at the stem or background.

# %% 
# Preprocess the sample image
image_path = "./fruits_subset/Apple_Rotten/rottenApple_2.jpg"

# Define the target category for explanation 
# (0 corresponds to 'fresh', 1 corresponds to 'rotten')
class_idx = 1

# Preprocess the sample image
img = helper_utils.preprocess_image(image_path, device)

# Compute CAM
heatmap = simplified_cam(
    model=fruits_model, 
    image_tensor=img, 
    class_idx=class_idx
)

# Display CAM
helper_utils.display_cam(img, heatmap)

# %% 5 - Comparison of Interpretability Techniques
# =============================================================================
# When to use each:
# 
# Feature Hierarchy: Use when you want to understand how a model learns, what patterns emerge at different depths, or debug training issues.
# 
# Saliency Maps: Use when you need precise pixel-level explanations, are concerned about adversarial vulnerabilities, or want to understand fine-grained sensitivities.
# 
# CAM/Grad-CAM: Use when you need human-interpretable region highlights, want to verify the model is looking at the right object, or need coarse localization for weakly-supervised tasks.
# 
# Pro Tip: Combine multiple techniques! For example:
# 
# Use CAM to verify the model focuses on the fruit (not background)
# Use Saliency to see exactly which pixels (e.g., specific spots, edges) drive the decision
# Use Feature Hierarchy to understand what low/mid/high-level features the model learned
# This multi-method approach gives you comprehensive understanding of your model's behavior.
# =============================================================================
