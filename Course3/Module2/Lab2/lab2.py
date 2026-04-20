# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:08:33 2026

C3 M2 L2 Saliency and Class Activation Map Visualization using PyTorch & ResNet

Deep learning models are often described as "black boxes". You input an image and get a prediction, but how does the model decide what it sees? The decision making process often remains opaque.

Saliency maps and Grad-CAM (Gradient weighted Class Activation Mapping) are two pivotal visualization techniques that help you "peek inside" the decision process of a model. They allow you to identify which areas or pixels in an image matter most to the network for a particular prediction.

In this lab, you will:

Load and preprocess images for a pre-trained ResNet model.
Perform predictions and retrieve class probabilities.
Compute saliency maps by backpropagating gradients from the target class all the way to the input pixels.
Compute class activation ("attention") maps (Grad-CAM) to visualize the spatial regions that influenced the choice of the model.
Overlay these visualizations on top of the original images to interpret the focus of the model.
Understand the intuition and mathematics behind each visualization, with step by step explanations of the code.

@author: ekirshin
"""
import os

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.transform import resize
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

import helper_utils

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% Loading the Pretrained ResNet50 and Class Labels
# You will utilize ResNet50, a powerful convolutional neural network trained on the ImageNet dataset (1000 classes).
# Saliency maps can be computed for any image classifier, but ImageNet models are excellent illustrations for this purpose.
# You should also fetch the class label names for making your outputs human readable.

# Load pretrained ResNet50 model and class labels from local cache
torch.hub.set_dir(os.path.join(os.getcwd(), 'pretrained_model'))
model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1).eval()
model.to(device)

# Class names
imagenet_class_mapping = tv_models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
print(f"Loaded {len(imagenet_class_mapping)} classes.")

# %% Image Preprocessing
# ResNet50 requires the following input specifications:

# Images as RGB, shape (224, 224)
# Pixel normalization using ImageNet statistics: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
# The following code demonstrates how you process an image from a file into a tensor:
# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
    
# %% Prediction and Visualizing Inputs
# You will now load an image, transform it, and obtain the top prediction from the model.
# You are encouraged to use your own images or try the examples provided below (place them in an images/ folder):
img_path = 'images/dog.jpg'  # Change to your own image path

if not os.path.exists(img_path):
    raise FileNotFoundError("Please provide a valid path to an image.")

img_pil = load_image(img_path)
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_prob, pred_class = torch.max(probs, dim=1)
    pred_label = imagenet_class_mapping[pred_class]
print(f"Predicted class: {pred_label} (prob: {pred_prob.item():.3f})\n")

# Show the image
plt.imshow(np.array(img_pil))
plt.title(f"Original Image\nPredicted: {pred_label}")
plt.axis('off')
plt.show()

# %% Saliency Maps
def compute_saliency_map(model, input_image, target_class=None):
    """
    Computes a saliency map for an input image using the gradients of the
    model's output with respect to the input pixels.

    Args:
        model: The neural network model used for classification.
        input_image: The input tensor image of shape [1, 3, H, W].
        target_class: The index of the target class for which the saliency
            map is computed. If None, the predicted class is used.

    Returns:
        saliency_map: A 2D numpy array representing pixel importance.
        pred_class: The index of the class used for the computation.
        pred_prob: The confidence score for the selected class.
    """
    # Clone the input and enable gradient tracking for the tensor
    input_image = input_image.clone().detach()
    input_image.requires_grad_()

    # Perform a forward pass to obtain the model logits
    output = model(input_image)

    # Calculate probabilities and determine the class for saliency analysis
    probs = torch.softmax(output, dim=1)
    if target_class is None:
        pred_prob, pred_class = torch.max(probs, dim=1)
        target_class = pred_class.item()
        pred_prob = pred_prob.item()
    else:
        pred_prob = probs[0, target_class].item()
        pred_class = target_class

    # Reset existing gradients in the model parameters
    model.zero_grad()

    # Execute backward pass to find gradients of the target class score
    output[0, target_class].backward()

    # Extract gradients of the output with respect to the input image
    gradients = input_image.grad.data[0]

    # Reduce the color channels by taking the absolute sum for a 2D map
    saliency_map = torch.abs(gradients).sum(dim=0).cpu().numpy()

    # Rescale the saliency map values to a range between 0 and 1
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min() + 1e-8)

    return saliency_map, pred_class, pred_prob

# %%
saliency_map, pred_class, pred_prob = compute_saliency_map(model, img_tensor)
print("saliency_map shape:", saliency_map.shape)
print("pred_class:", pred_class)
print("pred_prob:", pred_prob)
print("Class label:", imagenet_class_mapping[pred_class])

# %% Compute and Visualize the Saliency Map
def visualize_saliency(img_display, saliency_map, pred_class, pred_score, title):
    """
    Displays the original image, an enhanced saliency map, and an overlay.

    Args:
        img_display: Original image as a numpy array in uint8 format.
        saliency_map: Computed saliency array with values in range [0, 1].
        pred_class: Predicted class label or index for the image.
        pred_score: Numerical confidence score for the prediction.
        title: String identifier used for the plot title.

    Returns:
        None. Displays a three-panel matplotlib figure.
    """
    # Initialize a figure with three subplots for side-by-side visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Display the original input image in the first panel
    ax1.imshow(img_display)
    ax1.set_title(f'Original Image: {title}', fontsize=14)
    ax1.axis('off')

    # Apply gamma correction to enhance the contrast of the saliency map
    gamma = 0.7
    saliency_map_enhanced = np.power(saliency_map, gamma)

    # Determine dimensions and resize saliency map to match the source image
    h, w = img_display.shape[:2]
    saliency_map_resized = resize(
        saliency_map_enhanced, (h, w),
        order=1, mode='reflect', anti_aliasing=True
    )

    # Plot the enhanced saliency heatmap in the second panel
    saliency_heatmap = ax2.imshow(saliency_map_resized, cmap='inferno')
    ax2.set_title('Enhanced Saliency Map', fontsize=14)
    ax2.axis('off')
    fig.colorbar(saliency_heatmap, ax=ax2, fraction=0.046, pad=0.04)

    # Map the saliency values to the inferno colormap for RGB representation
    heatmap = cm.inferno(saliency_map_resized)[..., :3]
    # Normalize the display image and apply a fade factor for the background
    img_normalized = img_display / 255.0
    fade_factor = 0.3
    img_faded = img_normalized * fade_factor

    # Convert the faded image to grayscale to emphasize the saliency colors
    img_gray = np.mean(img_faded, axis=2, keepdims=True)
    img_gray = np.repeat(img_gray, 3, axis=2)

    # Define the alpha transparency and weight for the saliency overlay
    alpha = saliency_map_resized[:, :, np.newaxis]
    saliency_weight = 0.9

    # Blend the grayscale background with the colored heatmap
    overlay = (1 - alpha * saliency_weight) * img_gray + (
        alpha * saliency_weight) * heatmap
    # Clip values to ensure the final image stays within the valid [0, 1] range
    overlay = np.clip(overlay, 0, 1)

    # Render the combined overlay in the third panel with prediction details
    ax3.imshow(overlay)
    ax3.set_title(
        f'Saliency Overlay\nPrediction: {pred_class}\n'
        f'Confidence: {pred_score:.2f}', fontsize=14
    )
    ax3.axis('off')

    # Adjust layout to prevent overlapping and render the plots
    plt.tight_layout()
    plt.show()

# %%
# Reload your original (untransformed) image for display
img_display = np.array(img_pil)

# Compute saliency map
sal_map, pred_class_index, pred_prob = compute_saliency_map(model, img_tensor)

# Show results!
visualize_saliency(
    img_display,
    sal_map,
    imagenet_class_mapping[pred_class_index],
    pred_prob,
    title=os.path.splitext(os.path.basename(img_path))[0]
)
