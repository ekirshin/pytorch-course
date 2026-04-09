#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 A1
Created on Sat Apr  4 16:13:33 2026

@author: ek
"""
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
#import unittests

import torch
import torchvision.utils as vutils
from IPython.display import display
from torchvision import transforms
import torchinfo
import copy

import helper_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Preparing the Data Pipeline
# Compose the transformations for training: resize, augment, then preprocess
train_transform = transforms.Compose([
    # Resize images to a consistent square size (64x64 pixels)
    transforms.Resize((64, 64)),
    # Apply random horizontal flipping for data augmentation
    transforms.RandomHorizontalFlip(),
    # Apply random rotation (up to 10 degrees) for data augmentation
    transforms.RandomRotation(10),
    # Convert PIL images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize tensor values to range [-1, 1] (using mean=0.5, std=0.5)
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


# For validation: only resize and preprocess (no augmentation)
val_transform = transforms.Compose([
    # Resize images to the same consistent square size (64x64 pixels)
    transforms.Resize((64, 64)),
    # Convert PIL images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize tensor values to range [-1, 1] (using mean=0.5, std=0.5)
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

# %% # Define the path to the root directory containing the dataset
dataset_path = "./clothing-dataset-small"

# %% # Load the training and validation datasets
train_dataset, validation_dataset = helper_utils.load_datasets(
    # Path to the dataset directory
    dataset_path=dataset_path,
    # Apply the defined training transformations
    train_transform=train_transform,
    # Apply the defined validation transformations
    val_transform=val_transform,
    )

# %% # Get the list of class names automatically inferred from the folder structure
classes = train_dataset.classes

# Get the total number of classes
num_classes = len(classes)

# Print the discovered class names
print(f"Classes: {classes}")
# Print the total count of classes
print(f"Number of classes: {num_classes}")


# %% # Display a grid of sample images from the training dataset with their labels
helper_utils.show_sample_images(train_dataset)

# %% # Create DataLoaders for managing batching and shuffling
train_loader, val_loader = helper_utils.create_dataloaders(
    # Pass the training dataset
    train_dataset=train_dataset,
    # Pass the validation dataset
    validation_dataset=validation_dataset,
    # Define the number of images per batch
    batch_size=32
)

# %% 1.3 - Architecting the Classifier: Efficiency with Inverted Residuals
# InvertedResidualBlock
class InvertedResidualBlock(nn.Module):
    """
    Implements an inverted residual block, often used in architectures like MobileNetV2.
    
    This block features an expansion phase (1x1 convolution), a depthwise
    convolution (3x3 convolution), and a projection phase (1x1 convolution).
    It utilizes a residual connection between the input and the output of the projection.
    """
    
    def __init__(
        self, in_channels, out_channels, stride, expansion_factor, shortcut=None
    ):
        """
        Initializes the InvertedResidualBlock module.

        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels in the output tensor.
            stride (int): The stride to be used in the depthwise convolution.
            expansion_factor (int): The factor by which to expand the input channels
                                    in the expansion phase.
            shortcut: An optional module to be used for the shortcut connection,
                      typically to match dimensions if the stride is > 1 or
                      if channel counts differ.
        """
        # Initialize the parent nn.Module
        super().__init__()
        # Calculate the number of channels for the intermediate (expanded) representation
        # The hidden dimension is the expanded number of channels
        hidden_dim = in_channels * expansion_factor


        ### START CODE HERE ###

        # Define the expansion phase, which increases channel dimension
        # Expansion phase: increases the number of channels
        self.expand = nn.Sequential(
            # 1x1 pointwise convolution
            nn.Conv2d(in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1, stride=1, bias=False),
            # Batch normalization
            nn.BatchNorm2d(num_features=hidden_dim),
            # ReLU activation
            nn.ReLU(inplace=True),
        ) 

        ### END CODE HERE ###

        # Define the depthwise convolution phase
        # Depthwise convolution: lightweight spatial convolution per channel
        self.depthwise = nn.Sequential(
            # 3x3 depthwise convolution
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            # Batch normalization
            nn.BatchNorm2d(num_features=hidden_dim),
            # ReLU activation
            nn.ReLU(inplace=True),
        )

        ### START CODE HERE ###

        # Define the projection phase, which reduces channel dimension
        # Projection phase: reduces the number of channels to out_channels
        self.project = nn.Sequential(
            # 1x1 pointwise convolution (linear)
            nn.Conv2d(in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1, stride=1, bias=False),
            # Batch normalization
            nn.BatchNorm2d(num_features=out_channels),
        ) 

        ### END CODE HERE ###

        # Store the provided shortcut module
        # Optional shortcut connection for residual learning
        self.shortcut = shortcut

    def forward(self, x):
        """
        Defines the forward pass of the InvertedResidualBlock.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the block operations
                          and the residual connection, followed by a ReLU activation.
        """

        ### START CODE HERE ###
        
        # Store the original input for the residual connection
        # Save input for residual connection
        skip = x

        # Apply the expansion phase
        # Forward pass through the block
        # Expand channels
        out = self.expand(x)
        
        # Apply the depthwise convolution
        # Apply depthwise convolution
        out = self.depthwise(out)
        
        # Apply the projection phase
        # Project back to out_channels
        out = self.project(out)
        
        ### END CODE HERE ###

        # Check if a separate shortcut module is defined
        # If shortcut exists (for matching dimensions), use it
        # DO NOT REMOVE `None` from the `if` condition
        if self.shortcut is not None:
            
        ### START CODE HERE ###
            
            # Apply the shortcut module to the original input
            # Use the shortcut connection to match dimensions
            skip = self.shortcut(x)

        # Add the (potentially transformed) input (skip connection) to the output
        # Add the skip connection
        out = out + skip

        ### END CODE HERE ###      
        
        # Apply the final ReLU activation
        return F.relu(out)

# %% # --- Verification ---
# Define parameters for a sample block instance
batch_size=32
in_ch = 16 # Input channels
out_ch = 16 # Output channels (same for stride=1)
stride = 1
exp_factor = 3 # Expansion factor
img_size = 32 # Input image height/width

# Instantiate the block
block = InvertedResidualBlock(
    in_channels=in_ch,
    out_channels=out_ch,
    stride=stride,
    expansion_factor=exp_factor,
)

# Define the input tensor shape
input_size = (batch_size, in_ch, img_size, img_size)

# Configuration for torchinfo summary
config = {
    "input_size": input_size,
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": 3 # Show layers up to 3 levels deep
}

# Generate the summary
summary = torchinfo.summary(
    model=block,
    input_size=config["input_size"],
    col_names=config["attr_names"],
    depth=config["depth"]
)

# Display the formatted summary
print("--- Block Summary (Stride=1, Same Channels) ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_inv_res.html")

# %% MobileNetBackbone
# GRADED CLASS: MobileNetBackbone

class MobileNetBackbone(nn.Module):
    """
    Implements a simplified MobileNet-like backbone feature extractor.

    This class defines the initial stem and a sequence of inverted residual blocks
    to extract features from an input image.
    """

    def __init__(self):
        """
        Initializes the layers of the MobileNet backbone.
        """
        # Call the parent class (nn.Module) constructor
        super().__init__()
        # Define the initial "stem" convolution layer
        # This layer reduces spatial size and increases channel depth
        self.stem = nn.Sequential(
            # 3x3 convolution with stride 2
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),  # 3 input channels (RGB), 16 output
            # Apply batch normalization
            nn.BatchNorm2d(16),
            # Apply ReLU activation
            nn.ReLU(inplace=True),
        )

        ### START CODE HERE ###

        # Define the main stack of custom MobileNet-like blocks
        self.blocks = nn.Sequential(  
            # Each block progressively increases channels and reduces spatial dimensions
            # Create the first block
            self._make_block(in_channels=16, out_channels=24, stride=2, expansion_factor=3),
            # Create the second block
            self._make_block(in_channels=24, out_channels=32, stride=2, expansion_factor=3),
            # Create the third block
            self._make_block(in_channels=32, out_channels=64, stride=2, expansion_factor=6),
        )  

        ### END CODE HERE ###

    def _make_block(self, in_channels, out_channels, stride=1, expansion_factor=6):
        """
        Helper method to create a single InvertedResidualBlock.

        Arguments:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: The stride to be used in the depthwise convolution.
            expansion_factor: The factor to expand the channels internally.
        """

        ### START CODE HERE ###

        # Determine if a shortcut connection is needed
        # A shortcut is needed if input/output channels differ or if stride > 1
        condition = stride > 1 or in_channels != out_channels
        # If a shortcut is needed
        if condition:

        ### END CODE HERE ###

            # Define the shortcut connection
            shortcut = nn.Sequential(
                # 1x1 convolution to match dimensions and apply stride
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                # Apply batch normalization
                nn.BatchNorm2d(out_channels),
            )

        else:
            # No shortcut connection is needed
            shortcut = None

        ### START CODE HERE ###

        # Instantiate the InvertedResidualBlock
        block = InvertedResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expansion_factor=expansion_factor,
            shortcut=shortcut
        )

        ### END CODE HERE ###
        
        # Return the created block
        return block

    def forward(self, x):
        """
        Defines the forward pass of the backbone.

        Arguments:
            x: The input tensor (e.g., a batch of images).

        Returns:
            The output feature map tensor.
        """
        # Pass the input through the initial stem layer
        x = self.stem(x)
        # Pass the result through the main stack of blocks
        x = self.blocks(x)

        # Return the final feature map
        return x

# %% # --- Verification ---
# Define parameters for verification
batch_size=32
img_size = 64 # Input image height/width
depth = 3 # Summary depth

# Instantiate the backbone
backbone = MobileNetBackbone()

# Define the input tensor shape
input_size = (batch_size, 3, img_size, img_size)

# Configuration for torchinfo summary
config = {
    "input_size": input_size,
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": depth
}

# Generate the summary
summary = torchinfo.summary(
    model=backbone,
    input_size=config["input_size"],
    col_names=config["attr_names"],
    depth=config["depth"]
)

# Display the formatted summary
print("--- Backbone Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_backbone.html")

# %% Assembling the Full Classifier
class MobileNetLikeClassifier(nn.Module):
    """
    A classifier model that combines a feature extraction
    backbone with a simple classification head.
    """
    
    def __init__(self, num_classes=10):
        """
        Initializes the classifier components.

        Args:
            num_classes (int): The number of output classes for the final
                               classification layer.
        """
        # Initialize the parent nn.Module
        super().__init__()

        # Backbone extracts features from input images
        self.backbone = MobileNetBackbone()

        # Head processes the features to produce class predictions
        self.head = nn.Sequential(
            # Reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d(1),
            # Flatten the features into a 1D vector
            nn.Flatten(),
            # Map the flattened features to the number of output classes
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Defines the forward pass of the classifier.

        Args:
            x: The input tensor (batch of images).

        Returns:
            torch.Tensor: The raw, unnormalized output scores (logits)
                          for each class.
        """
        # Pass the input through the feature extraction backbone
        x = self.backbone(x)
        # Pass the features through the classification head
        x = self.head(x)
        # Return the final output
        return x

# %% # Ensure num_classes matches the number of categories in your dataset
mobilenet_classifier = MobileNetLikeClassifier(num_classes=num_classes)

# %% Verification
# Define parameters for verification
batch_size=32
img_size = 64 # Input image height/width
depth = 3 # Summary depth

# Define the input tensor shape
input_size = (batch_size, 3, img_size, img_size)

# Configuration for torchinfo summary
config = {
    "input_size": input_size,
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": depth # Show layers up to 3 levels deep for detail
}

# Generate the summary for the complete classifier
summary = torchinfo.summary(
    model=mobilenet_classifier,
    input_size=config["input_size"],
    col_names=config["attr_names"],
    depth=config["depth"]
)

# Display the formatted summary
print("--- Classifier Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model.html")

# %% Training the Classifier
# Your efficient MobileNetLikeClassifier is fully assembled! Now it's time to teach it how to distinguish between different fashion items. This involves setting up the essential components for the training process:

# Class Weights: Real-world datasets are often imbalanced, meaning some classes have many more examples than others. Training directly on such data can bias the model towards the majority classes. To counteract this, you'll calculate class weights. These weights give more importance to under-represented classes during loss calculation, encouraging the model to learn them effectively.

# Loss Function: Measures how far the model's predictions are from the true labels. CrossEntropyLoss is suitable for this multi-class classification task, and you'll configure it to use the calculated class weights.

# Optimizer: Adjusts the model's weights based on the loss to improve performance. Adam is a popular and effective choice.

# Learning Rate Scheduler: Dynamically adjusts the learning rate during training. A StepLR scheduler will decrease the learning rate periodically, which can help the model converge more effectively.

# First, calculate the class weights based on the distribution of samples in your training dataset. Then, define the weighted CrossEntropyLoss function.

# Calculate class weights to handle imbalance in the dataset
class_weights = helper_utils.compute_class_weights(train_dataset)

# Move the weights tensor to the correct device (e.g., 'cuda' or 'cpu')
class_weights = class_weights.to(device)

# Define the loss function for multi-class classification, incorporating the calculated class weights
loss_fcn = nn.CrossEntropyLoss(weight=class_weights)

# Print the calculated weights for verification
print("Calculated class weights:")
for i, weight in enumerate(class_weights):
    print(f"- Class '{classes[i]}': {weight:.4f}")
    
# %%
# Define the Adam optimizer, passing the model's parameters and initial learning rate
optimizer = torch.optim.Adam(mobilenet_classifier.parameters(), lr=0.01)

# Define a learning rate scheduler that reduces the LR by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# %% # Set the number of epochs for training
n_epochs = 20

# %% # Start the training loop
trained_classifier =  helper_utils.training_loop(
    mobilenet_classifier, 
    train_loader, 
    val_loader, 
    loss_fcn, 
    optimizer, 
    scheduler, 
    device, 
    n_epochs=n_epochs
)

# %% Evaluating the Classifier
# Display predictions
helper_utils.display_random_predictions_per_class(trained_classifier, val_loader, classes, device)

# %% Building a Visual Search Engine
# Teaching Similarity: The Triplet Dataset

class TripleDataset(Dataset):
    """
    A custom Dataset class that returns triplets of images (anchor, positive, negative).

    This class wraps a standard dataset and, for a given index, returns the 
    item at that index (anchor), a random item with the same label (positive),
    and a random item with a different label (negative).
    """
    
    def __init__(self, dataset):
        """
        Initializes the TripleDataset.

        Args:
            dataset: The base dataset (e.g., torchvision.datasets) 
                     which contains (data, label) pairs.
        """

        # Store the original dataset
        self.dataset = dataset

        # Get a list of all available labels
        self.labels = range(len(dataset.classes))

        # Create a mapping from labels to their corresponding indices in the dataset
        self.labels_to_indices = self._get_labels_to_indices()

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        # The length is the same as the original wrapped dataset
        return len(self.dataset)

    def _get_labels_to_indices(self):
        """
        Creates a dictionary mapping each label to a list of indices.
        
        Returns:
            A dictionary where keys are labels and values are lists of 
            indices in the dataset that have that label.
        """
        # Initialize an empty dictionary
        labels_to_indices = {}
        # Iterate over the entire dataset
        for idx, (_, label) in enumerate(self.dataset):
            # If the label is not yet in the dictionary, add it with an empty list
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            # Append the current index to the list for its label
            labels_to_indices[label].append(idx)
        # Return the completed map
        return labels_to_indices
    

    def _get_positive_negative_indices(self, anchor_label):
        """
        Finds random indices for a positive and a negative sample.

        Args:
            anchor_label: The label of the anchor sample.

        Returns:
            A tuple (positive_index, negative_index).
        """

        ### START CODE HERE ###

        # Get all indices for the anchor label
        positive_indices = self.labels_to_indices[anchor_label]
        # Randomly select one index from the list of positive indices
        positive_index = random.choice(positive_indices)

        # Get all indices for a negative label
        # Randomly choose a label that is different from the anchor label
        negative_label = random.choice([label for label in self.labels if label != anchor_label]) 
        
        # Get all indices for the chosen negative label
        negative_indices = self.labels_to_indices[negative_label]
        # Randomly select one index from the list of negative indices
        negative_index = random.choice(negative_indices)

        ### END CODE HERE ###

        return positive_index, negative_index

    def __getitem__(self, idx):
        """
        Retrieves a triplet (anchor, positive, negative) for a given index.

        Args:
            idx: The index of the anchor item.

        Returns:
            A tuple containing the anchor image, positive image, and negative image.
        """

        ### START CODE HERE ###

        # Get the anchor image and label
        anchor_image, anchor_label = self.dataset[idx]

        # Get positive and negative indices based on the anchor label
        positive_index, negative_index = self._get_positive_negative_indices(anchor_label)

        # Get a positive image (same label)
        positive_image, _ = self.dataset[positive_index]

        # Get a negative image (different label)
        negative_image, _ = self.dataset[negative_index]

        ### END CODE HERE ###

        return (anchor_image, positive_image, negative_image)
    
    
