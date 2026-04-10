#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 L1b
Created on Wed Mar  4 21:57:49 2026

@author: ek
"""
import glob
import os
import random
from collections import defaultdict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import helper_utils
import training_functions

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Use Case 2: Tracking Environmental Change
# Define the path to the root directory containing the change dataset.
change_data_dir = './levir_cd_plus_simulated/'

# %% # Scan the data directory and display the statistical summary.
helper_utils.display_change_dataset_stats(change_data_dir)

# %%
# Display a side-by-side comparison of pairs of each class from the dataset.
helper_utils.display_random_change_pairs(change_data_dir)

# %%
class SiameseNetwork(nn.Module):
    """
    A flexible Siamese Network that can process either image triplets or pairs.

    This network uses a shared backbone (embedding network) to generate feature
    vectors (embeddings) for multiple input images simultaneously. It can operate
    in two modes: one for training with triplets (anchor, positive, negative)
    and one for inference with pairs.

    Attributes:
        embedding_network (nn.Module): The shared backbone network.
    """
    def __init__(self, embedding_network):
        """
        Initializes the Siamese Network.
        
        Args:
            embedding_network (nn.Module): The backbone network that generates embeddings.
        """
        # Initialize the parent nn.Module class.
        super().__init__()
        # Store the shared backbone model.
        self.embedding_network = embedding_network
        
    def forward(self, *inputs, triplet_bool=True):
        """
        Processes either a triplet or a pair of images through the embedding network.

        Args:
            *inputs: A sequence of input tensors.
                     - If triplet_bool is True, expects (anchor, positive, negative).
                     - If triplet_bool is False, expects (image1, image2).
            triplet_bool (bool): If True, operates in triplet mode for training.
                                 If False, operates in pair mode for inference.
        
        Returns:
            tuple: A tuple of output embedding tensors.
        """
        if triplet_bool:
            # Handle the case for training with triplets.
            if len(inputs) != 3:
                raise ValueError("In triplet mode, expected 3 inputs: anchor, positive, negative.")
            
            # Unpack the triplet inputs.
            anchor, positive, negative = inputs
            
            # Generate embeddings for each image using the shared backbone.
            anchor_output = self.embedding_network(anchor)
            positive_output = self.embedding_network(positive)
            negative_output = self.embedding_network(negative)
            
            return anchor_output, positive_output, negative_output
        
        else:
            # Handle the case for inference with image pairs.
            if len(inputs) != 2:
                raise ValueError("In pair mode, expected 2 inputs: before_img, after_img.")
            
            # Unpack the pair inputs.
            img1, img2 = inputs
            
            # Generate embeddings for both images using the shared backbone.
            output1 = self.embedding_network(img1)
            output2 = self.embedding_network(img2)
            
            return output1, output2
    
    def get_embedding(self, image):
        """
        Generates a single embedding for a given image.
        
        Args:
            image (torch.Tensor): A single image tensor.

        Returns:
            torch.Tensor: The resulting embedding vector.
        """
        # Pass the single image through the backbone to get its embedding.
        return self.embedding_network(image)

# %%
class ChangeDetectionDataset(Dataset):
    """
    A PyTorch Dataset for loading 'Before' and 'After' image pairs
    for a change detection task.

    This class scans a directory where subdirectories are named
    'Positive', 'Negative', and 'No_Change', each containing 'Before' and
    'After' subfolders with corresponding image pairs.
    """
    def __init__(self, base_dir, transform=None):
        """
        Initializes the dataset by scanning the data directory and organizing file paths.
        
        Args:
            base_dir (str): Path to the root directory which contains the
                            'Positive', 'Negative', and 'No_Change' folders.
            transform (callable, optional): PyTorch transforms to be applied to each image.
        """
        self.base_dir = base_dir
        self.transform = transform
        
        # Define a mapping from class names to integer labels for convenience.
        self.class_to_label = {'Positive': 0, 'Negative': 1, 'No_Change': 2}
        
        # Build the complete list of all available image pairs from the source directory.
        self.image_pairs = self._create_image_pairs()
        
        # Raise an error if the dataset directory is empty or improperly structured.
        if not self.image_pairs:
            raise RuntimeError(f"No valid image pairs found in {base_dir}. Check directory structure.")

    def _create_image_pairs(self):
        """Scans the directory to build a list of (before_path, after_path, label) tuples (one-time setup)."""
        image_pairs = []
        # Iterate through each change category ('Positive', 'Negative', 'No_Change').
        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(self.base_dir, class_name)
            before_dir = os.path.join(class_dir, 'Before')
            after_dir = os.path.join(class_dir, 'After')
            
            # Skip this category if its 'Before' directory does not exist.
            if not os.path.isdir(before_dir):
                continue

            # Iterate through all files in the 'Before' directory.
            for filename in os.listdir(before_dir):
                if filename.lower().endswith(('.png', '.jpg')):
                    # Construct the full paths for the 'Before' and corresponding 'After' images.
                    before_path = os.path.join(before_dir, filename)
                    after_path = os.path.join(after_dir, filename)
                    
                    # Add the pair to the list only if the corresponding 'After' image exists.
                    if os.path.exists(after_path):
                        image_pairs.append((before_path, after_path, label))
        return image_pairs

    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Generates and returns one pair of images and its corresponding label.
        
        Args:
            idx (int): The index of the image pair to retrieve from the dataset.
            
        Returns:
            tuple: A tuple containing (before_img, after_img, label).
        """
        # Retrieve the file paths and label for the requested index.
        before_path, after_path, label = self.image_pairs[idx]
        
        # Load the 'before' and 'after' images from their respective paths.
        before_img = self._load_image(before_path)
        after_img = self._load_image(after_path)
            
        return before_img, after_img, label
        
    def _load_image(self, path):
        """
        Helper function to robustly load a single image from a given path.

        Args:
            path (str): The file path of the image to load.

        Returns:
            The loaded and transformed image, typically a torch.Tensor.
        """
        # Use a context manager to ensure the file is properly closed after loading.
        with Image.open(path) as img:
            # Ensure the image is in RGB format, as many networks expect 3 channels.
            image = img.convert("RGB")
            # Apply any specified transformations (e.g., resizing, tensor conversion).
            if self.transform:
                image = self.transform(image)
        return image

# %%
# Initialize the full dataset object. 
full_change_dataset = ChangeDetectionDataset(change_data_dir)

# %% Preparing the Change Data for Training
# ImageNet normalization statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformations for the training set (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Transformations for validation set (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# %%
# Split the full dataset into training and validation sets.
train_dataset, val_dataset = helper_utils.create_change_datasets_splits(
    full_dataset=full_change_dataset,
    train_split=0.8, 
    train_transform=train_transform,
    val_transform=val_transform
)

# Create a DataLoader for the training set.
train_dataloader_change = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Create a DataLoader for the validation set.
val_dataloader_change = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the final "virtual" size of each dataset split.
print(f"Total training pairs: {len(train_dataset)}")
print(f"Total validation pairs: {len(val_dataset)}")

# %%
# Visualize a random pair from the training dataloader.
helper_utils.show_random_pair(train_dataloader_change)

# %%
# Create the EfficientNet-based embedding network
efficientnet_embedding = helper_utils.get_efficientnet_embedding_backbone(embedding_dim=128)

# %%
# Instantiate the main Siamese model, using the EfficientNet network as its base
siamese_efficientnet = SiameseNetwork(embedding_network=efficientnet_embedding)

# %% Addressing class imbalance issue
# Compute the class weights using the full and train datasets.
class_weights = helper_utils.compute_change_class_weights(
    train_dataset=train_dataset,
    full_untransformed_dataset=full_change_dataset
)

# Print class weights
label_to_class = {v: k for k, v in full_change_dataset.class_to_label.items()}
for i, weight in enumerate(class_weights):
    print(f"Class '{label_to_class[i]}': Weight = {weight:.4f}")

# %% Define Weighted Contrastive Loss
class WeightedContrastiveLoss(nn.Module):
    """
    A contrastive loss function that incorporates class weights to handle imbalance.
    
    It adapts a multi-class problem into a binary similarity problem where
    'No_Change' is 'similar' and 'Positive'/'Negative' are 'dissimilar'.
    """
    def __init__(self, device, margin=1.0, class_weights=None):
        """
        Initializes the weighted contrastive loss function.
        
        Args:
            device (torch.device): The device to move class weights to.
            margin (float): The margin for dissimilar pairs.
            class_weights (torch.Tensor, optional): A tensor of weights for each class.
                                                      Shape: (num_classes,).
        """
        super().__init__()
        self.margin = margin
        self.device = device
        
        # Move weights to the correct device once during initialization for efficiency.
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None

    def forward(self, output1, output2, label):
        """
        Computes the weighted contrastive loss for a batch of embeddings.

        Args:
            output1 (torch.Tensor): Embeddings for the first set of images.
            output2 (torch.Tensor): Embeddings for the second set of images.
            label (torch.Tensor): The multi-class labels (0, 1, or 2) from the dataset.
        """
        # Calculate the pairwise Euclidean distance between the output embeddings.
        distances = F.pairwise_distance(output1, output2)
        
        # Convert multi-class labels (0, 1, 2) to binary similarity labels (1, 1, 0).
        # A label of 2 ('No_Change') is considered similar (0), others are dissimilar (1).
        binary_label = (label != 2).float()

        # Calculate the contrastive loss for each sample in the batch.
        loss_per_sample = (
            # Loss for similar pairs aims to minimize the distance.
            (1 - binary_label) * distances.pow(2) +
            # Loss for dissimilar pairs aims to make the distance larger than the margin.
            binary_label * torch.clamp(self.margin - distances, min=0).pow(2)
        )

        # Apply class-specific weights to the loss if they are provided.
        if self.class_weights is not None:
            # Gather the correct weight for each sample using its original multi-class label.
            weights = self.class_weights[label.long()]
            
            # Multiply each sample's loss by its corresponding class weight.
            loss_per_sample = loss_per_sample * weights
            
        # Return the mean of the (potentially weighted) losses for the batch.
        return loss_per_sample.mean()

# %%
# Initialize the custom weighted contrastive loss function with the calculated class weights.
contrastive_loss = WeightedContrastiveLoss(margin=2.0, class_weights=class_weights, device=device)

# %% # Initialize the AdamW optimizer for the new EfficientNet-based model
optimizer_change = optim.AdamW(siamese_efficientnet.parameters(), lr=1e-3)

# Initialize the new, more flexible scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_change,
    mode='min',      # Reduce LR when the validation loss stops decreasing
    factor=0.2,      # New LR = LR * factor
    patience=2,      # Wait 2 epochs with no improvement before reducing LR
)

# %%
# Run the training process
trained_efficientnet = training_functions.training_loop_change(
    # The Siamese network model built on EfficientNet.
    model=siamese_efficientnet,
    # DataLoader for the training set.
    train_loader=train_dataloader_change,
    # DataLoader for the validation set.
    val_loader=val_dataloader_change,
    # The weighted contrastive loss function.
    loss_fcn=contrastive_loss,
    # The optimizer for updating model weights.
    optimizer=optimizer_change,
    # The learning rate scheduler.
    scheduler=scheduler,
    # The compute device (e.g., 'cpu' or 'cuda').
    device=device,
    # File path to save the model with the lowest validation loss.
    save_path='./saved_models/best_change_siamese.pth',
    # The total number of epochs for training.
    n_epochs=10
)

# %%
# Evaluate the model that was trained
optimal_threshold = training_functions.evaluation_loop(
    # Your fine-tuned EfficientNet Siamese model.
    model=trained_efficientnet,
    # The DataLoader for the change detection validation set.
    data_loader=val_dataloader_change,
    # The same weighted contrastive loss used during training.
    loss_fcn=contrastive_loss,
    # The compute device (e.g., 'cpu' or 'cuda').
    device=device
)

# %%
# Call the function with your trained model, validation loader, and optimal threshold
helper_utils.plot_confusion_matrix_and_metrics(
    model=trained_efficientnet,
    data_loader=val_dataloader_change,
    threshold=optimal_threshold,
    device=device
)

# %%
# Visualize the model's performance on a few random samples from the validation set.
helper_utils.show_change_val_predictions(
    model=trained_efficientnet,
    val_loader=val_dataloader_change,
    model_threshold=optimal_threshold,
    device=device
)

#%% Additional testing
# Define the path to the "before" image
before_image = "./change_samples/Positive/Before/train_803.png"

# Define the path to the "after" image
after_image = "./change_samples/Positive/After/train_803.png"

# Visualize the model's performance on a unseen sample pairs.
helper_utils.predict_greenery_change(
    model=trained_efficientnet, 
    before_path=before_image, 
    after_path=after_image, 
    model_threshold=optimal_threshold, 
    transform=val_transform, 
    device=device
)