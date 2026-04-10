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
n_epochs = 10

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
    
# %% # --- Verification Cell ---
# Set seed for reproducibility of random sampling within the dataset
random.seed(42)

# Create a copy of the validation dataset to use as a base for the toy triplet dataset
toy_dataset_base = copy.deepcopy(validation_dataset)

# Apply a simple transformation
toy_dataset_base.transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Instantiate the TripleDataset using the prepared base dataset
triple_dataset_toy = TripleDataset(dataset=toy_dataset_base)

# Retrieve the first triplet (index 0) from the dataset
anchor_img, positive_image, negative_image = triple_dataset_toy[0]

# Display the triplet
images_list = [anchor_img, positive_image, negative_image]
grid = vutils.make_grid(images_list, nrow=3, padding=2) # nrow=3 ensures side-by-side display
grid_pil = transforms.ToPILImage()(grid)
print("Sample Triplet (Anchor, Positive, Negative):")
display(grid_pil)
    
# %% # Instantiate the TripleDataset using the main training dataset
triple_dataset = TripleDataset(train_dataset)

# Create a DataLoader for the TripleDataset
siamese_dataloader = torch.utils.data.DataLoader(
    triple_dataset,
    batch_size=32,  # Define the number of triplets per batch
    shuffle=True,
)

# %% Architecting the Visual Search Model
# With your TripleDataset ready to feed comparison examples, you can now build the Siamese Network itself. This architecture cleverly reuses the feature extraction capabilities you already built for the classifier, demonstrating the power of modular design.
# 2.3.1 - The Siamese Encoder: Reusing the Backbone
# The core of the Siamese Network is the encoder. This is the part that takes an image and converts it into a meaningful numerical representation (an embedding). Instead of building a new encoder from scratch, you'll leverage the MobileNetBackbone you created earlier. This is a prime example of parameter sharing and modularity in action – the backbone learns general visual features, which are useful for both classification and similarity tasks.
# You'll wrap this backbone in a SiameseEncoder class. This wrapper adds a simple "representation head" consisting of an AdaptiveAvgPool2d layer followed by a Flatten layer. This head takes the 2D feature map produced by the backbone and converts it into a fixed size 1D vector – the final embedding.
# The SiameseEncoder class is defined below. Notice how it takes a backbone as input during initialization.
class SiameseEncoder(nn.Module):
    """
    Implements an encoder module suitable for Siamese networks.

    This class takes a pre-defined backbone (feature extractor) and adds a
    representation head (pooling + flatten) to produce a fixed-size vector
    embedding for an input image.
    """

    def __init__(self, backbone):
        """
        Initializes the SiameseEncoder.

        Args:
            backbone (nn.Module): The convolutional neural network to use
                                  as the feature extractor.
        """
        # Initialize the parent nn.Module
        super(SiameseEncoder, self).__init__()

        # Store the provided backbone model
        self.backbone = backbone

        # Define the representation head
        self.representation = nn.Sequential(
            # Apply adaptive average pooling to reduce spatial dimensions to 1x1
            # This makes the output size independent of the input image size (after backbone)
            nn.AdaptiveAvgPool2d(1),
            # Flatten the 1x1 feature map into a 1D vector
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Defines the forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of images).

        Returns:
            torch.Tensor: The final 1D embedding vector for the input.
        """
        # 1. Extract features using the backbone
        features = self.backbone(x)
        # 2. Convert feature map to a fixed-size vector using the representation head
        representation = self.representation(features)
        # Return the resulting embedding vector
        return representation

# %% Instantiate the SiameseEncoder. You'll pass the backbone, trained_classifier.backbone from your previously created mobilenet_classifier to demonstrate reuse. This is efficient – the backbone's learned features are directly transferred.
# Create the Siamese Encoder instance
# Reuse the backbone from the classifier model you built earlier!
siamese_encoder = SiameseEncoder(
    backbone=trained_classifier.backbone # Pass the existing backbone
)

# %% The Siamese Network Wrapper
# Now you create the main SiameseNetwork class. This acts as a wrapper around your siamese_encoder. Its primary role during training is to take the three images of a triplet (anchor, positive, negative), pass each of them through the same siamese_encoder instance (ensuring shared weights), and return the three resulting embeddings. It also includes a get_embedding method, which is useful for processing single images during inference (when you want to find similar items).
class SiameseNetwork(nn.Module):
    """
    Implements the main Siamese Network structure.

    This network takes multiple inputs (anchor, positive, negative during training)
    and processes each through a shared `embedding_network` (the SiameseEncoder)
    to produce corresponding embeddings.
    """

    def __init__(self, embedding_network):
        """
        Initializes the SiameseNetwork.

        Args:
            embedding_network (nn.Module): The shared encoder network (e.g., SiameseEncoder)
                                           that generates embeddings from images.
        """
        # Initialize the parent nn.Module
        super(SiameseNetwork, self).__init__()
        # Store the shared embedding network
        self.embedding_network = embedding_network

    def forward(self, anchor, positive, negative):
        """
        Defines the forward pass for training with image triplets.

        Args:
            anchor (torch.Tensor): The batch of anchor images.
            positive (torch.Tensor): The batch of positive images (same class as anchor).
            negative (torch.Tensor): The batch of negative images (different class from anchor).

        Returns:
            tuple: A tuple containing the embeddings for anchor, positive, and negative images.
                   (anchor_output, positive_output, negative_output)
        """
        # Process the anchor image through the embedding network
        anchor_output = self.embedding_network(anchor)
        # Process the positive image through the *same* embedding network (shared weights)
        positive_output = self.embedding_network(positive)
        # Process the negative image through the *same* embedding network (shared weights)
        negative_output = self.embedding_network(negative)

        # Return the generated embeddings
        return anchor_output, positive_output, negative_output

    def get_embedding(self, image):
        """
        Generates an embedding for a single input image. Used for inference/retrieval.

        Args:
            image (torch.Tensor): The input image tensor (should include batch dimension).

        Returns:
            torch.Tensor: The embedding vector for the image.
        """
        # Pass the single image through the embedding network
        return self.embedding_network(image)
# %% # Instantiate the Siamese Network
siamese_network = SiameseNetwork(embedding_network=siamese_encoder)
    
# %% Training the Siamese Network
# You have the data generator (TripleDataset) and the model architecture (SiameseNetwork wrapping the SiameseEncoder). Now, set up the final pieces needed for training:

# Loss Function: You'll use nn.TripletMarginLoss. This loss function directly implements the core idea of Siamese training: it calculates the distances between the anchor positive pair (
# ) and the anchor negative pair (
# ) and penalizes the model if 
#  is not smaller than 
#  by at least a specified margin. The goal is 
# .

# Optimizer: torch.optim.AdamW is a good choice for optimizing the network's weights.
# EDITABLE CELL: Feel free to play around with different "margin" and "lr" values

# Define the Triplet Margin Loss function
# margin=1.0: Enforces that the negative sample should be at least 1.0 distance unit farther than the positive sample
# p=2.0: Use Euclidean distance (L2 norm)
loss_fcn = nn.TripletMarginLoss(margin=1.0, p=2.0)

# Define the AdamW optimizer, passing the Siamese network's parameters and a learning rate
optimizer = torch.optim.AdamW(siamese_network.parameters(), lr=0.001)

# %%
# Now, train the Siamese network using the siamese_training_loop helper function. This loop focuses solely on minimizing the TripletMarginLoss. Unlike classification, the goal here isn't to achieve a specific accuracy during training, but rather to organize the embedding space effectively by pulling similar items (anchor, positive) closer and pushing dissimilar items (anchor, negative) apart. The decreasing loss value is your main indicator of progress. The true test of the model's success will be its performance in the visual search task later. Train for a small number of epochs initially to verify the setup.
# Define the number of training epochs
num_epochs = 15

# %%
# Run the training loop for the Siamese network
helper_utils.siamese_training_loop(
    # The Siamese model instance
    model=siamese_network,
    # The DataLoader providing triplets
    dataloader=siamese_dataloader,
    # The TripletMarginLoss function
    loss_fcn=loss_fcn,
    # The AdamW optimizer
    optimizer=optimizer,
    # The compute device
    device=device,
    # The number of epochs to train
    n_epochs=num_epochs,
)

# %% Performing Visual Search (Retrieval)
# With a trained Siamese network, you now have an encoder (siamese_encoder) capable of turning fashion images into meaningful embedding vectors. The final step is to use these embeddings to find similar items. This process typically involves:

# Selecting a Query Image: Choose an image for which you want to find similar items.
# Generating Query Embedding: Pass the query image through the trained siamese_encoder to get its embedding vector.
# Generating Catalog Embeddings: Process all images in your product catalog (represented here by the validation_dataset) through the siamese_encoder to create an embedding for each item. In a real system, these would be pre calculated and stored in a database for fast lookup.
# Calculating Distances: Compute the distance (e.g., Euclidean distance) between the query embedding and all catalog embeddings.
# Ranking: Sort the catalog items based on their distance to the query, from smallest (most similar) to largest (least similar).
# Retrieving Top Results: Select the top N items with the smallest distances.
# Selecting the Query Image
# You have two options for selecting the image you want to search for:

# Use a Provided Sample Image: You can use one of the sample images available in the ./images/ directory. Here are the paths:
# ./images/dress.jpg
# ./images/hat.jpg
# ./images/longsleeve.jpg
# ./images/pant.jpg
# ./images/shoes.jpg
# ./images/shorts.jpg
# ./images/t_shirt.jpg
# Upload Your Own Image: Use the widget below to upload a custom JPG image.
# Running helper_utils.upload_jpg_widget() displays an upload widget.
# You can only upload images with a .jpg extension.
# Each image file size must not exceed 5 MB.
# After successful upload, the widget will display the file path (e.g., ./uploads/your_image.jpg). Copy this path.
# You can reuse the widget multiple times without rerunning the cell.
# Run the cell below to display the image upload widget.
# Display the widget for uploading JPG images

#helper_utils.upload_jpg_widget()

# %%
# Replace the example path below with the path to your desired query image
# Example using a provided image: image_path = './images/hat.jpg'
# Example using an uploaded image: image_path = './uploads/your_uploaded_image.jpg'

image_path = './images/t_shirt.jpg' ### Add your image path here

# %% # Load the selected image using the helper function
query_img = helper_utils.get_query_img(image_path)
# Display the query image
print("Query Image:")
display(query_img)

# %%  Define a function get_query_img_embedding to process a single PIL image: apply the necessary transformations, pass it through the trained encoder, and return its embedding as a NumPy array.
def get_query_img_embedding(encoder, transform, img, device):
    """
    Generates an embedding vector for a single query PIL image.

    Args:
        encoder (nn.Module): The trained embedding model (e.g., SiameseEncoder).
        transform (callable): The torchvision transforms to apply (e.g., resize, normalize).
        img (PIL.Image): The input query image.
        device (torch.device): The device ('cuda' or 'cpu') to perform inference on.

    Returns:
        np.ndarray: The embedding vector as a NumPy array.
    """
    # Apply the transformations (resize, ToTensor, normalize)
    tensor_img = transform(img)

    # Add a batch dimension (B, C, H, W) as the model expects batches
    # and move the tensor to the specified device
    query_img_tensor = tensor_img.unsqueeze(0).to(device)

    # Set the encoder to evaluation mode (important for layers like BatchNorm, Dropout)
    encoder.eval()
    # Perform inference without calculating gradients to save memory and computation
    with torch.no_grad():
        # Pass the image tensor through the encoder model
        query_img_embedding = encoder(query_img_tensor)

    # Move the resulting embedding tensor from the device (e.g., GPU) back to the CPU
    # and convert it into a NumPy array for easier handling (e.g., distance calculations)
    query_img_embedding_np = query_img_embedding.cpu().numpy()
    # Return the embedding as a NumPy array
    return query_img_embedding_np

# %% # Generate the embedding for the sample query image
query_img_embedding = get_query_img_embedding(siamese_encoder, val_transform, query_img, device)

# Print the shape of the resulting embedding vector (should be [1, embedding_dim])
print("Shape of query image embedding:", query_img_embedding.shape)

# %% Now, generate embeddings for all images in your "catalog" (using the validation_dataset here as a stand in for a full product catalog). The get_embeddings helper function efficiently processes the entire dataset.
# Use the validation dataset as the "catalog" of items to search within
catalog = validation_dataset

# Use a helper function to efficiently generate embeddings for all items in the catalog
print("Generating embeddings for the catalog...")
embeddings = helper_utils.get_embeddings(siamese_encoder, catalog, device)
print(f"Generated {len(embeddings)} embeddings for the catalog.")

# %% Finding and Displaying Similar Items
# With embeddings generated for the query and the entire catalog, use the find_closest helper function to identify the indices of the num_samples most similar images in the catalog based on Euclidean distance.
# EDITABLE CELL: # Define how many similar items to retrieve

num_samples = 5

# %% # Use a helper function to find the indices of the items in the catalog
# whose embeddings are closest (smallest Euclidean distance) to the query embedding
print(f"Finding the top {num_samples} closest items...")
closest_indices = helper_utils.find_closest(embeddings, query_img_embedding, num_samples)
print("Indices of closest items:", closest_indices)

# %% Finally, retrieve the actual images and their labels from the catalog using the closest_indices and display them. These are the top visual matches for your query image according to your trained Siamese network!
# Loop through the indices of the closest images found
print(f"\nDisplaying the {num_samples} most similar items found in the catalog:")
for idx_c in closest_indices:
    # Retrieve the image and its true label from the catalog dataset using the index
    # The helper function likely handles converting tensor back to displayable format
    img_c, label_idx_c = helper_utils.get_image(catalog, idx_c) # Assuming get_image returns PIL + label index
    label_c = catalog.classes[label_idx_c] # Get class name from index
    # Print the class label of the retrieved image
    print(f"Retrieved Item - Class: {label_c} (Index: {idx_c})")
    # Display the retrieved image
    display(img_c)

