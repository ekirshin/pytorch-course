#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 L1
Created on Wed Feb 25 22:07:28 2026

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

# %% Use Case 1: Signature Verification

# Define the path to the root directory containing the signature dataset.
signature_data_dir = './Signature_Verification_v5_v11/'

# Scan the data directory and display the statistical summary.
helper_utils.display_signature_dataset_summary(signature_data_dir)

#%% Display a side-by-side comparison of a random real and fake signature from the dataset.
helper_utils.display_random_signature_pair(signature_data_dir)

# %%
class SignatureTripletDataset(Dataset):
    """
    A PyTorch Dataset for creating signature triplets for verification.

    This class scans a directory of real and fake signatures, organized by
    user ID, and generates triplets (anchor, positive, negative) on the fly
    for training a Siamese network with triplet loss.
    """
    
    def __init__(self, base_data_dir, triplets_per_user=100, transform=None):
        """
        Initializes the dataset by scanning the data directory and organizing file paths.
        
        Args:
            base_data_dir (str): The root directory of the signature dataset.
            triplets_per_user (int): The "virtual" number of triplets to generate
                                     per individual for one epoch.
            transform (callable, optional): PyTorch transforms to be applied to each image.
        """
        self.base_data_dir = base_data_dir
        self.triplets_per_user = triplets_per_user
        self.transform = transform
        # Build the map of all available image paths from the source directory.
        self.signature_map = self._create_signature_map()
        # Create the definitive list of individuals to be used in the dataset.
        self.user_ids = list(self.signature_map.keys())
        
        # Raise an error if the dataset directory is empty or improperly structured.
        if not self.user_ids:
            raise RuntimeError(f"No valid individuals found in {base_data_dir}. Check directory structure and image counts.")

    def _create_signature_map(self):
        """Scans the directory to build a map of individuals to their signature paths (one-time setup)."""
        # Define paths for real and fake signature directories.
        real_signatures_dir = os.path.join(self.base_data_dir, 'Real')
        fake_signatures_dir = os.path.join(self.base_data_dir, 'Fake')
        signature_map = defaultdict(lambda: {'real': [], 'fake': []})

        # Validate that the 'Real' signatures directory exists.
        if not os.path.isdir(real_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {real_signatures_dir}")

        # Validate that the 'Fake' signatures directory exists.
        if not os.path.isdir(fake_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {fake_signatures_dir}")

        # Iterate through each user ID directory.
        all_ids = sorted(os.listdir(real_signatures_dir))
        for user_id in all_ids:
            if user_id.startswith('ID_'):
                # Find all real and fake signature images for the current user.
                real_images = glob.glob(os.path.join(real_signatures_dir, user_id, '*.jpg'))
                fake_images = glob.glob(os.path.join(fake_signatures_dir, user_id, '*.jpg'))
                
                # Only include individuals with enough images to create a valid triplet.
                if len(real_images) >= 2 and len(fake_images) >= 1:
                    signature_map[user_id]['real'] = real_images
                    signature_map[user_id]['fake'] = fake_images
                    
        return signature_map

    def __len__(self):
        """
        Returns the "virtual" length of the dataset for an epoch.
        
        This is not the total number of possible triplets, but a fixed number
        to define the size of an epoch.
        """
        return len(self.user_ids) * self.triplets_per_user

    def __getitem__(self, index):
        """
        Generates and returns one triplet of images on the fly.
        
        Args:
            index (int): Required by PyTorch's Dataset API but not used here,
                         as triplets are generated randomly.
            
        Returns:
            tuple: A tuple containing the (anchor, positive, negative) image tensors.
        """
        # Randomly select an individual to form the triplet.
        person_id = random.choice(self.user_ids)
        
        # Sample two distinct real images for the anchor and positive samples.
        anchor_path, positive_path = random.sample(self.signature_map[person_id]['real'], 2)
        # Sample one fake image for the negative sample.
        negative_path = random.choice(self.signature_map[person_id]['fake'])

        # Load images from paths and apply any specified transformations.
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)
        
        return (anchor_img, positive_img, negative_img)

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
full_signature_dataset = SignatureTripletDataset(signature_data_dir)

# %%
# Pre-calculated mean and standard deviation for this dataset
mean = [0.861, 0.861, 0.861]
std = [0.274, 0.274, 0.274]

# Transformations for the training set (with augmentation)
train_transform = transforms.Compose([
    # Randomly apply slight affine transformations (shear and translation)
    # This mimics variations in writing slant and position
    transforms.RandomAffine(degrees=0, shear=10, translate=(0.1, 0.1)),
    # Randomly apply a slight perspective shift
    # This can simulate viewing the signature from a different angle
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
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
train_dataset, val_dataset = helper_utils.create_signature_datasets_splits(
    full_dataset=full_signature_dataset,
    train_split=0.8, 
    train_transform=train_transform,
    val_transform=val_transform
)

# Create a DataLoader for the training set.
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Create a DataLoader for the validation set.
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the final "virtual" size of each dataset split.
print(f"Total training triplets:    {len(train_dataset)}")
print(f"Total validation triplets:  {len(val_dataset)}")

# %%
# Visualize a random triplet from the training dataloader.
helper_utils.show_random_triplet(train_dataloader)

# %% Constructing the Siamese Network
class SimpleEmbeddingNetwork(nn.Module):
    """
    A simple Convolutional Neural Network to generate a fixed-size embedding from an image.
    This network is designed for 224x224 RGB input images.

    Attributes:
        conv (nn.Sequential): The convolutional layers for feature extraction.
        fc (nn.Sequential): The fully connected layers for generating the embedding.
    """
    def __init__(self, embedding_dim=128):
        # Initialize the parent nn.Module class.
        super(SimpleEmbeddingNetwork, self).__init__()
        
        # Define the convolutional layers that act as a feature extractor.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            # Add a dropout layer for regularization to prevent overfitting.
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            # Add another dropout layer.
            nn.Dropout(0.4),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, stride=2)
        )
        
        # Define the fully connected layers that produce the final embedding vector.
        self.fc = nn.Sequential(
            # The input size is derived from the output of the final conv layer.
            nn.Linear(128 * 25 * 25, 256), nn.ReLU(),
            # Use a dropout layer with a higher rate for stronger regularization.
            nn.Dropout(0.6),
            # The final linear layer maps the features to the desired embedding dimension.
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input batch of images.

        Returns:
            torch.Tensor: The output embedding vector for each image in the batch.
        """
        # Pass the input through the convolutional feature extractor.
        x = self.conv(x)
        # Flatten the 3D feature map into a 1D vector for each item in the batch.
        x = x.view(x.size(0), -1) 
        # Pass the flattened vector through the fully connected layers.
        x = self.fc(x)
        # Return the final embedding.
        return x
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
# Define the desired size for the final embedding vector
embedding_dim = 128

# Create an instance of the base model that generates embeddings
embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)

# Create the main Siamese network model, using the embedding network
siamese_network = SiameseNetwork(embedding_network=embedding_net)    

# %%
# Initialize the Triplet Margin Loss function
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

# %%
# Initialize the AdamW optimizer to update the model's weights
optimizer_siamese = optim.AdamW(siamese_network.parameters(), lr=1e-3)

# Set step_size=2 and gamma=0.1 to decrease the LR by a factor of 10 every 2 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer_siamese, step_size=2, gamma=0.1)

# %%
# Define the distance threshold for validation accuracy calculation.
threshold_dist = 0.8

# Execute the main training and validation loop.
trained_siamese = training_functions.training_loop_signature(
    # The Siamese network model instance.
    model=siamese_network,
    # DataLoader for the training set.
    train_loader=train_dataloader,
    # DataLoader for the validation set.
    val_loader=val_dataloader,
    # The triplet margin loss function.
    loss_fcn=triplet_loss,
    # The optimizer for updating model weights.
    optimizer=optimizer_siamese,
    # The learning rate scheduler.
    scheduler=scheduler,
    # The distance threshold for validation accuracy.
    threshold=threshold_dist,
    # The compute device (e.g., 'cpu' or 'cuda').
    device=device,
    # File path to save the best performing model.
    save_path='./saved_models/best_signature_siamese.pth',
    # The total number of epochs for training.
    n_epochs=5
)

# %%
# Visualize the model's performance on a few random triplets from the validation set.
helper_utils.show_signature_val_predictions(
    trained_siamese,
    val_dataloader,
    threshold=threshold_dist,
    device=device
)

# %% Real-World Application: One-Shot Signature Verification
# Define the path to the real signature image
signature_anchor = "./signature_samples/Real/real_6_4.jpg"

# Define the path to the signature image to verify
signature_to_verify = "./signature_samples/Fake/fake_6_3.jpg"

# %% Perform one-shot verification on the two images
helper_utils.verify_signature(
    model=trained_siamese, 
    genuine_path=signature_anchor, 
    test_path=signature_to_verify, 
    threshold=threshold_dist,
    transform=val_transform, 
    device=device
)

#%% Use Case 2: Tracking Environmental Change
