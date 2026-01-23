#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:43:08 2026

@author: ek
"""

import os

from IPython.display import Image as DisplayImage
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision.io import decode_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm

import helper_utils

#%% 
# Check if the OxfordIIITPet data folder exists
ox3_pet_data_path = './oxford3pet_data'
if os.path.exists(ox3_pet_data_path) and os.path.isdir(ox3_pet_data_path):
    ox3_pet_download = False  # Data folder exists, will be loaded from
else:
    ox3_pet_download = True  # Data folder doesn't exist, will be downloaded
    
#%% # Load an image
image = Image.open('./images/mangoes.jpg')

# Dimensions of the original PIL image
print("Original PIL Image Dimensions:", image.size)
print(f"The maximum pixel value is: {image.getextrema()[0][1]}, and the minimum is: {image.getextrema()[0][0]}")

#%% # Convert the PIL image to a PyTorch Tensor
img_tensor = transforms.ToTensor()(image)

# Dimensions (shape) of the tensor
# [C, H, W] format
print(f"Dimensions After Converting to a Tensor: {img_tensor.shape}")
print(f"The maximum pixel value is: {img_tensor.max()}, and the minimum is: {img_tensor.min()}")

#%% # Convert the tensor back to a PIL image
img_pil = transforms.ToPILImage()(img_tensor)

# Dimensions of the converted back PIL image
print("Dimensions After Converting Back to PIL:", img_pil.size)

#%% # Visualize the original and converted images
helper_utils.show_images([image, img_pil], titles=("Original Image", "After PIL→Tensor→PIL conversion"))

#%% # Define the path to the image file.
image_path = './images/apples.jpg'

# Load the image
image = decode_image(image_path)

print(f"Image tensor dimensions: {image.shape}")
print(f"Image tensor dtype: {image.dtype}")
print(f"The maximum pixel value is: {image.max()}, and the minimum is: {image.min()}\n")

#%% # Use the DisplayImage to render the image
DisplayImage(image_path, width=500, height=500)

#%% # Create a batch of images (./images/ contains only 6 images). The images are loaded as 300x300 pixels
images_tensor = helper_utils.load_images("./images/")

# The size is 6 images x 3 color channels x 300 pixels height x 300 pixels width
print(f"Image tensor dimensions: {images_tensor.shape}")

#%% # Make a grid from the loaded images (2 rows of 3 for 6 images)
grid = vutils.make_grid(tensor=images_tensor, nrow=3, padding=5, normalize=True)

# the shape comes from 
# num_images/nrow*pixel_height+(num_images/nrow+1)*padding = 2*300+3*5 = 615
# nrow*pixel_width+(nrow+1)*padding = 3*300+4*5 = 920
print(f"Image tensor dimensions: {grid.shape}")
print(f"The maximum pixel value is: {grid.max()}, and the minimum is: {grid.min()}\n")

# Display the grid of images using a helper function
helper_utils.display_grid(grid)

#%% # Define the path to save the image file.
image_path = "./fruits_grid.png"

# Save the grid as a PNG image
vutils.save_image(tensor=grid, fp=image_path)

#%% # Use the DisplayImage to render the image
DisplayImage(image_path)

#%% Transformations
original_image = Image.open('./images/strawberries.jpg')

#%% # Define the resize transformation (50x50 square)
resize_transform = transforms.Resize(size=50)

# Apply the transformation
resized_image = resize_transform(original_image)

# (Width, Height)
print(f"Original Dimensions: {original_image.size}")
print(f"Resized Dimensions:  {resized_image.size}\n")

helper_utils.show_images(
    images=[original_image, resized_image], 
    titles=("Original", "Resized to (50, 50)")
)

#%% # Define the center crop transformation (256x256)
center_crop_transform = transforms.CenterCrop(size=256)

# Apply the transformation
cropped_image = center_crop_transform(original_image)

# (Width, Height)
print(f"Original Dimensions: {original_image.size}")
print(f"Cropped Dimensions:  {cropped_image.size}\n")

helper_utils.show_images(
    images=[original_image, cropped_image],
    titles=("Original", "Center Crop (256, 256)")
)

#%% # Define the RandomResizedCrop transformation (224x224)
random_resized_crop_transform = transforms.RandomResizedCrop(size=224)

# Apply the transformation
cropped_resized_image_1 = random_resized_crop_transform(original_image)
cropped_resized_image_2 = random_resized_crop_transform(original_image)
cropped_resized_image_3 = random_resized_crop_transform(original_image)

# (Width, Height)
print(f"Original Dimensions: {original_image.size}")
print(f"RandomResizedCrop 1 Dimensions:  {cropped_resized_image_1.size}")
print(f"RandomResizedCrop 2 Dimensions:  {cropped_resized_image_2.size}")
print(f"RandomResizedCrop 3 Dimensions:  {cropped_resized_image_3.size}\n")

helper_utils.show_images(
    images=[original_image, cropped_resized_image_1],
    titles=("Original (2048, 2048)", "RandomResizedCrop 1 (224, 224)")
)
helper_utils.show_images(
    images=[cropped_resized_image_2, cropped_resized_image_3],
    titles=("RandomResizedCrop 2 (224, 224)", "RandomResizedCrop 3 (224, 224)")
)

#%% RandomHorizontalFlip
# Define the horizontal flip transformation
# Set p=1.0 to guarantee the flip happens for this demonstration
flip_transform = transforms.RandomHorizontalFlip(p=1.0)

# Apply the transformation
flipped_image = flip_transform(original_image)

helper_utils.show_images(
    images=[original_image, flipped_image],
    titles=("Original", "RandomHorizontalFlip (p=1.0)")
)

#%% ColorJitter
# Define the ColorJitter transformation
# The values determine the random range for each property.
jitter_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

# Apply the transformation
jittered_image = jitter_transform(original_image)

helper_utils.show_images(
    images=[original_image, jittered_image],
    titles=("Original", "ColorJitter")
)

#%% Custom transformations
class SaltAndPepperNoise:
    """
    A custom transform to add salt and pepper noise to a PIL image.

    Args:
        salt_vs_pepper (float): The ratio of salt to pepper noise.
                                (e.g., 0.5 is an equal amount of each).
        amount (float): The total proportion of pixels to be affected by noise.
    """
    def __init__(self, salt_vs_pepper=0.5, amount=0.04):
        self.s_vs_p = salt_vs_pepper
        self.amount = amount

    def __call__(self, image):
        # Make a copy of the image
        output = np.copy(np.array(image))

        # Add Salt Noise
        num_salt = np.ceil(self.amount * image.size[0] * image.size[1] * self.s_vs_p)
        # Generate random coordinates for salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.size]
        # Set pixels to white
        output[coords[1], coords[0]] = 255  

        # Add Pepper Noise
        num_pepper = np.ceil(self.amount * image.size[0] * image.size[1] * (1.0 - self.s_vs_p))
        # Generate random coordinates for pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.size]
        # Set pixels to black
        output[coords[1], coords[0]] = 0

        # Convert the NumPy array back to a PIL image
        return Image.fromarray(output)

    def __repr__(self):
        return self.__class__.__name__ + f'(salt_vs_pepper={self.s_vs_p}, amount={self.amount})'

#%%    
# Instantiate your custom transformation
sp_transform = SaltAndPepperNoise(salt_vs_pepper=0.5, amount=0.5)

# Apply the transformation
sp_image = sp_transform(original_image)

#%% 
helper_utils.show_images(
    images=[original_image, sp_image],
    titles=("Original", "With Salt & Pepper Noise")
)

#%% Normalize
# Convert to tensor (scales to [0, 1])
tensor_image = transforms.ToTensor()(original_image)

# Define the normalization transform using ImageNet stats
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Apply the transformation
normalized_tensor = normalize_transform(tensor_image)

# Visualize the distribution before and after normalization
helper_utils.plot_histogram(tensor_image, normalized_tensor, "Comparison of Pixel Distribution Before and After Normalization")

#%% Mean and STD calculation of a dataset
def calculate_mean_std(dataset):
    """
    Calculates the mean and standard deviation of a PyTorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset for which to
                                            calculate the stats. It should
                                            return image tensors.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing the mean and
                                      standard deviation tensors, each of
                                      shape (C,).
    """
    # Create a DataLoader to iterate through the dataset in batches for efficiency.
    # shuffle=False because the order of images doesn't matter for this calculation.
    loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize tensors to store the sum of pixel values for each (RGB) channel.
    channel_sum = torch.zeros(3)
    # Initialize tensors to store the sum of squared pixel values for each channel.
    channel_sum_sq = torch.zeros(3)
    # Initialize a counter for the total number of pixels.
    num_pixels = 0

    # Wrap the loader with tqdm to create a progress bar for monitoring.
    for images, _ in tqdm(loader, desc="Calculating Dataset Stats"):
        # Add the total number of pixels in this batch to the running total.
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        
        # Sum the pixel values across the batch, height, and width dimensions,
        # leaving only the channel dimension. Add this to the running total.
        channel_sum += images.sum(dim=[0, 2, 3])
        
        # Square each pixel value, then sum them up similarly to the step above.
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    # Calculate the mean for each channel.
    mean = channel_sum / num_pixels
    # Calculate the standard deviation using the formula: sqrt(E[X^2] - E[X]^2)
    std = (channel_sum_sq / num_pixels - mean ** 2) ** 0.5

    # Return the calculated mean and standard deviation.
    return mean, std 

#%% Simple transform pipeline to compute mean and STD
# Define a simple transformation
simple_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Load the OxfordIIITPet dataset, applying the simple transform to each image
my_dataset = datasets.OxfordIIITPet(root=ox3_pet_data_path,
                                    split='test',                 # Specify using the test set
                                    download=ox3_pet_download,    # Download if not already present
                                    transform=simple_transform    # Apply the defined transformations
                                   )

# Compute the mean and standard deviation for the dataset
dataset_mean, dataset_std = calculate_mean_std(my_dataset)

print(f"\nCalculation Complete.")
print(f"Dataset Mean: {dataset_mean}")
print(f"Dataset Std:  {dataset_std}")

#%% 
# A simple transform to get a clean, un-augmented version of the images
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    # Skip normalization to keep the image's pixel values in a display-friendly range.
])

# The full augmentation pipeline with all random transformations
full_augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    SaltAndPepperNoise(amount=0.001),
    transforms.ToTensor(),
    # Using `mean` and `std` values as calculated on the 100x100 images
    transforms.Normalize(mean=dataset_mean,
                         std=dataset_std)
])

#%% A Dataset Without Augmentations
# Load the dataset with ONLY the base transforms
original_dataset = datasets.OxfordIIITPet(root=ox3_pet_data_path, 
                                          split='test',
                                          download=ox3_pet_download,
                                          transform=base_transform
                                         )

# Create a DataLoader for the original images
original_loader = data.DataLoader(original_dataset, batch_size=9, shuffle=True)

# Get one fixed batch of original images
original_images, _ = next(iter(original_loader))

# Create a grid from the batch of images, arranging them with 3 images per row.
grid = vutils.make_grid(original_images, nrow=3, padding=2) 

print("Original Un-augmented Batch:\n")
helper_utils.display_grid(grid)

#%% Applying augmentation pipeline
# Use a loop to apply different random augmentations
for i in range(3):
    
    augmented_batch = []
    
    # Loop through each original image in the fixed batch
    for img_tensor in original_images:
        
        # Convert tensor back to PIL image to apply random transforms
        img_pil = transforms.ToPILImage()(img_tensor)

        # Apply the random augmentation pipeline
        augmented_tensor = full_augmentation_pipeline(img_pil)

        # Add the augmented tensor to the list for display
        augmented_batch.append(augmented_tensor)

    # Stack the list of augmented tensors into a single batch tensor
    final_batch = torch.stack(augmented_batch)

    # Create a grid from the batch of images, arranging them with 3 images per row
    grid = vutils.make_grid(final_batch, nrow=3, padding=2)
    
    print(f"\nAugmented Batch - Run #{i + 1}")
    helper_utils.display_grid(grid)