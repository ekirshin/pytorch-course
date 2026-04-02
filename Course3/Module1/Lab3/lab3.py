#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 L3
Created on Mon Mar 30 20:38:17 2026

@author: ek
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchinfo

import helper_utils

# Set seed
SEED = 42

# %%
# Setup device priority: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

# %%
# Define the path to the root directory where the image dataset is stored.
dataset_path = "./UCMerced_LandUse/Images/"

# Create an easy to read list of class names for use in plots and labels.
class_names = ['Agricultural', 'Baseball Diamond', 'Buildings', 'Dense Residential',
               'Harbor', 'Medium Residential', 'Mobile Home Park', 'Parking Lot',
               'Runway', 'Sparse Residential', 'Storage Tanks', 'Tennis Court', 
               'Airplane', 'Beach', 'Chaparral', 'Forest', 'Freeway', 'Golf Course',
               'Intersection', 'Overpass', 'River'
              ]

# %% Assemble Your Data Pipeline
# Define your pipelines of transformations for training and validation data.
# Use the pre-calculated mean and std of this dataset.
# Pre-calculated mean and std of this dataset
mean = [0.485, 0.490, 0.451]
std = [0.214, 0.197, 0.191]

# Transformations for the training set (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Transformations for validation set (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])

# %% # Create the training and validation datasets by splitting the main dataset.
train_dataset, val_dataset = helper_utils.create_datasets(
    dataset_path, 
    train_transform,
    val_transform,
    train_split=0.8,
    seed=SEED
)

# Determine the number of unique classes from the dataset's properties.
num_classes = len(train_dataset.classes)

# Print a summary of the dataset split.
print(f"Total Number of Classes:  {num_classes}")     
print(f"Training set size:        {len(train_dataset)}")
print(f"Validation set size:      {len(val_dataset)}")

# %% # Define the number of images to process in each batch.
batch_size = 32

# Create the training and validation DataLoaders using the helper function.
train_loader, val_loader = helper_utils.create_dataloaders(train_dataset, val_dataset, batch_size)

# %% Visualize Training Samples
# Display the sample images from train set
helper_utils.show_sample_images(train_dataset, class_names)

# %% DenseLayer: The Engine of Feature Reuse
class DenseLayer(nn.Module):
    """A single dense layer module as described in the DenseNet architecture.

    This layer implements the bottleneck design, where a 1x1 convolution reduces
    the number of feature maps before a 3x3 convolution is applied. The output
    feature maps are then concatenated with the input feature maps.

    Args:
        in_channels (int): The number of **input channels**.
        growth_rate (int): The number of feature maps to produce (**k** in the paper).
        bn_size (int): The multiplicative factor for the number of bottleneck channels.
    """
    def __init__(self, in_channels, growth_rate=32, bn_size=4):
        super(DenseLayer, self).__init__()

        # Bottleneck layer: 1x1 convolution for dimensionality reduction.
        self.dimension_reduction = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
            ),
        )

        # Feature extraction layer: 3x3 convolution to generate new features.
        self.feature_extraction = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        """Defines the forward pass of the dense layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after concatenating with the input.
        """
        # Pass the input through the bottleneck and feature extraction layers.
        new_features = self.dimension_reduction(x)
        new_features = self.feature_extraction(new_features)
        
        # Concatenate the new feature maps with the original input feature maps.
        concatenated_features = torch.cat((x, new_features), 1)

        return concatenated_features
# %% Verifying the Blueprint: The DenseLayer Summary
# Create an instance of the DenseLayer.
denselayer = DenseLayer(
    in_channels=3,      # Accepts an input with 3 channels (e.g., RGB).
    growth_rate=12,     # Will produce 12 new feature maps.
    bn_size=4           # The bottleneck layer will have (4 * 12) = 48 channels.
)

# Define the shape for a single image (Channels, Height, Width).
img_shape = (3, 64, 64)

# Define the full input shape for a batch of images.
input_size =  (batch_size, *img_shape)
# %%
# Define a configuration dictionary to store parameters for the model summary.
config = {
    "input_size": input_size,
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": 2
}

# Generate the model summary object using torchinfo with the specified configuration.
summary = torchinfo.summary(
    model=denselayer, 
    input_size=config["input_size"], 
    col_names=config["attr_names"], 
    depth=config["depth"]
)

# Display the summary as a styled HTML table.
print("--- Model Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_summary_layer.html")

# %% DenseBlock: Accumulating Knowledge
# The DenseBlock is the heart of the DenseNet. It acts as a smart container that stacks multiple DenseLayer-s, wiring them together in the signature "densely connected" pattern. If a DenseLayer is a single sentence contributing a new piece of information, the DenseBlock is the paragraph where these sentences are sequentially combined to build a complex idea.
# Within the block, the output of one layer, containing all prior features plus its new ones, becomes the direct input for the next. This leads to a systematic growth in the network's collective knowledge, making the feature maps progressively "thicker" as they flow through the block.
# __init__:
# This constructor's primary role is to build the stack of DenseLayers. It loops for num_layers and creates a new DenseLayer in each iteration.
# The key logic is in calculating the input channels for each new layer: in_channels + i * growth_rate. This formula is the mathematical heart of the block, ensuring that each DenseLayer is aware of the accumulated feature maps from all the layers that came before it.
# forward:
# The forward pass executes the information accumulation. Its logic is straightforward because the complex channel calculations were already handled in the constructor.
# Initialize Features: A tensor named features is initialized with the block's input, x.
# Sequentially Process: The code iterates through the stack of layers. In each step, the current features tensor is passed into a DenseLayer, which returns an even thicker tensor (with growth_rate new channels). This thicker tensor becomes the new features for the next iteration.
# Return Final Output: After the loop finishes, features holds the combined output of all layers in the block, which is then returned.
class DenseBlock(nn.Module):
    """A container for a sequence of DenseLayer modules.

    This class groups multiple DenseLayer instances to form a single "dense block"
    as described in the DenseNet architecture. Within the block, each layer
    receives the feature maps from all preceding layers as its input.

    Args:
        num_layers (int): The number of **DenseLayer** modules in the block.
        in_channels (int): The number of channels in the **input tensor**.
        growth_rate (int): The number of new channels produced by each DenseLayer.
        bn_size (int): The multiplicative factor for the bottleneck layer channels.
    """
    def __init__(self, num_layers, in_channels, growth_rate=32, bn_size=4):
        super(DenseBlock, self).__init__()

        # Initialize a module list to hold all layers in the block.
        self.layers = nn.ModuleList()

        # Sequentially add DenseLayer modules to the block.
        for i in range(num_layers):
            # The input channels for each new layer is the initial number of channels
            # plus the accumulated growth from all previous layers.
            layer = DenseLayer(
                in_channels + i * growth_rate, growth_rate, bn_size
            )
            self.layers.append(layer)

    def forward(self, x):
        """Defines the forward pass for the DenseBlock.

        Args:
            x (torch.Tensor): The input tensor for the block.

        Returns:
            torch.Tensor: The output tensor after passing through all layers.
        """
        # The 'features' tensor holds the concatenated outputs from all layers.
        features = x
        
        # Pass the features through each dense layer in the block.
        for layer in self.layers:
            features = layer(features)
            
        return features
# %% Tracing the Knowledge Accumulation
# Create an instance of the DenseBlock.
denseblock = DenseBlock(
    in_channels=3,      # The block accepts an input with 3 channels.
    growth_rate=12,     # Each DenseLayer within the block adds 12 channels.
    bn_size=4,          # The bottleneck multiplier used in each DenseLayer.
    num_layers=2,       # The block will contain 2 consecutive DenseLayers.
)

# %%
# Define a configuration dictionary to store parameters for the model summary.
config = {
    "input_size": input_size, # (batch_size, *img_shape)
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": 3
}

# Generate the model summary object using torchinfo with the specified configuration.
summary = torchinfo.summary(
    model=denseblock, 
    input_size=config["input_size"], 
    col_names=config["attr_names"], 
    depth=config["depth"]
)

# Display the summary as a styled HTML table.
print("--- Model Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_summary_block.html")

# %% TransitionLayer: The Efficient Bridge
# After a DenseBlock has diligently accumulated knowledge and created a large, detailed feature map, the TransitionLayer steps in to act as a crucial bridge to the next block. If a DenseBlock is a dense chapter packed with information, the TransitionLayer is the concise summary at the chapter's end, preparing the reader for the next part of the story.

# Its purpose is twofold: to compress the information (reduce the number of channels) to keep the model efficient, and to downsample (reduce the height and width), which helps the network learn more abstract, high-level patterns.

# __init__:

# The constructor builds a compact and efficient pipeline to perform both compression and downsampling in one go. It bundles these operations into a single nn.Sequential module.

# The first key piece of logic is calculating the number of output channels: out_channels = int(in_channels * compression_factor). This is the compression step, effectively summarizing the vast number of features from the preceding block into a more manageable set.

# It then defines the self.transition pipeline, which first stabilizes the incoming features with BatchNorm2d and ReLU, and then performs its core tasks:

# A 1x1 Conv2d layer to perform the actual channel reduction.

# An AvgPool2d layer with a stride of 2 to cut the feature map's height and width in half, achieving the downsampling.

# forward:

# The forward pass is incredibly straightforward, as all the complex operations were already packaged neatly into the self.transition module during initialization.

# Direct Execution: It takes the large tensor output from a DenseBlock and simply passes it through the pre-built transition pipeline.

# Return Final Output: The layer returns a new tensor that is both "thinner" (fewer channels) and smaller in spatial dimensions, perfectly prepared to be the input for the next DenseBlock.
class TransitionLayer(nn.Module):
    """A transition layer used between two dense blocks in a DenseNet.

    This layer is responsible for reducing the number of channels (compression)
    via a 1x1 convolution and downsampling the spatial dimensions of the
    feature maps using average pooling.

    Args:
        in_channels (int): The number of channels in the **input tensor**.
        compression_factor (float): A factor between 0 and 1 to reduce the
                                    number of feature maps.
    """
    def __init__(self, in_channels, compression_factor=0.5):
        super(TransitionLayer, self).__init__()

        # Determine the number of output channels after applying compression.
        out_channels = int(in_channels * compression_factor)

        # The layer consists of batch normalization, a 1x1 convolution for channel
        # reduction, and average pooling for spatial downsampling.
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Defines the forward pass for the transition layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The downsampled and compressed output tensor.
        """
        return self.transition(x)
# %% DenseNet: Assembling the Full Architecture
# The DenseNet class is the master architect that assembles all the building blocks you have seen, DenseLayer, DenseBlock, and TransitionLayer, into a complete, powerful network. This class is responsible for creating the network's stem, building the main body by correctly stacking and connecting the blocks, and adding the final classification head.

# __init__:

# This method acts as the master blueprint for the entire network. It doesn't contain the layer logic itself but calls helper methods to construct the three main parts of the model: the initial feature extractor (self.features), the main body of dense blocks and transitions (self.dense_blocks), and the final self.classifier.

# _get_initial_features:

# This helper creates the network's "stem." Its job is to perform an initial, aggressive downsampling of the input image. It uses a standard combination of a large-kernel convolution and max pooling to quickly reduce the spatial dimensions and create the first set of feature maps that will be fed into the main body of the network.

# _get_dense_blocks:

# This is the most critical helper method, responsible for building the network's "body." It programmatically constructs the repeating pattern of DenseBlock -> TransitionLayer.

# It iterates through the block_config tuple, which defines how many DenseLayer-s are in each DenseBlock.

# A key responsibility of this method is to track the number of feature maps. It calculates the channel growth after each DenseBlock and the channel compression after each TransitionLayer, feeding the correct in_channels to the next component.

# After building the main blocks, it appends the final layers (like BatchNorm2d and AdaptiveAvgPool2d) that form the "head" of the network, preparing the data for the final classification layer.

# forward:

# This method defines the end-to-end journey of the data. The flow is very clean and sequential:

# The input image first passes through the initial self.features block.

# The resulting feature maps are then passed sequentially through the entire list of self.dense_blocks (which contains the alternating DenseBlock-s and TransitionLayer-s).

# Finally, the output is flattened and passed to the self.classifier to produce the final predictions.    
class DenseNet(nn.Module):
    """
    Implements the DenseNet architecture.

    Args:
        growth_rate (int): The number of feature maps each layer adds.
        block_config (tuple of ints): The number of layers in each dense block.
        num_init_features (int): The number of filters in the initial convolutional layer.
        bn_size (int): The multiplicative factor for the number of bottleneck channels.
        compression_factor (float): The factor by which to reduce the number of channels 
                                    in transition layers.
        num_classes (int): The number of output classes for the final classifier.
    """
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression_factor=0.5,
        num_classes=1000,
    ):
        super(DenseNet, self).__init__()

        # Define the initial feature extraction block.
        self.features = self._get_initial_features(num_init_features)

        # Define the main body of the network with the corrected call.
        self.dense_blocks = self._get_dense_blocks(
            num_init_features,
            block_config,
            growth_rate,
            bn_size,
            compression_factor,
        )

        # Define the final fully connected classification layer.
        self.classifier = nn.Linear(self.num_features, num_classes)

    def _get_initial_features(self, num_init_features):
        """
        Creates the initial convolutional block for feature extraction and downsampling.
        
        Args:
            num_init_features (int): The number of output channels for the first convolution.
            
        Returns:
            nn.Sequential: A sequential container for the initial layers.
        """
        # Create a sequential module for the initial layers.
        convolution_block = nn.Sequential(
            # 7x7 convolution with stride 2 for initial downsampling.
            nn.Conv2d(
                3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
            ),
            # Batch normalization.
            nn.BatchNorm2d(num_init_features),
            # ReLU activation.
            nn.ReLU(inplace=True),
            # 3x3 max pooling with stride 2 for further downsampling.
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        return convolution_block


    def _get_dense_blocks(self, num_init_features, block_config, growth_rate, bn_size, compression_factor):
        """
        Constructs the sequence of dense blocks and transition layers.
        
        Args:
            num_init_features (int): The initial number of features.
            block_config (tuple of ints): The number of layers for each dense block.
            growth_rate (int): The growth rate for dense blocks.
            bn_size (int): The bottleneck size factor.
            compression_factor (float): The compression factor for transition layers.
            
        Returns:
            nn.ModuleList: A list containing the dense blocks, transition layers, 
                           and final classification head components.
        """
        # Create a ModuleList to store all network blocks.
        dense_blocks = nn.ModuleList()
        
        # Initialize the number of features.
        num_features = num_init_features

        # Iterate through the block configurations to build the network.
        for i, num_layers in enumerate(block_config):
            # Create a new DenseBlock.
            db = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            # Add the DenseBlock to the list.
            dense_blocks.append(db)

            # Update the number of features based on the growth rate.
            num_features = num_features + num_layers * growth_rate

            # Add a transition layer after each dense block, except for the last one.
            if i != len(block_config) - 1:
                transition = TransitionLayer(
                    in_channels=num_features, compression_factor=compression_factor
                )
                # Add the transition layer to the list.
                dense_blocks.append(transition)
                # Update the number of features after compression.
                num_features = int(num_features * compression_factor)

        # Add the final batch normalization layer.
        dense_blocks.append(nn.BatchNorm2d(num_features))
        # Add the final ReLU activation.
        dense_blocks.append(nn.ReLU(inplace=True))
        # Add global average pooling to create a fixed-size feature vector.
        dense_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Store the final number of features for the classifier layer.
        self.num_features = num_features
        return dense_blocks

    def forward(self, x):
        """
        Defines the forward pass of the DenseNet model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        # Pass the input through the initial feature extractor.
        x = self.features(x)

        # Pass the features through each block in the main body.
        for block in self.dense_blocks:
            x = block(x)

        # Flatten the output tensor for the fully connected layer.
        x = torch.flatten(x, 1)

        # Pass the flattened features through the classifier.
        x = self.classifier(x)

        return x
# %% Assembling the Final Blueprint
# Instantiate the DenseNet model with its default configuration.
densenet = DenseNet()
# %%
# Define a configuration dictionary to store parameters for the model summary.
config = {
    "input_size": input_size, # (batch_size, *img_shape)
    "attr_names": ["input_size", "output_size", "num_params"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #"],
    "depth": 2
}

# Generate the model summary object using torchinfo with the specified configuration.
summary = torchinfo.summary(
    model=densenet, 
    input_size=config["input_size"], 
    col_names=config["attr_names"], 
    depth=config["depth"]
)

# Display the summary as a styled HTML table.
print("--- Model Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_summary.html")

# %% Initialize the Training Configurations
# Set the random seed to ensure that model is always initialized with the same random weights.
torch.manual_seed(SEED)

# Create an instance of your custom DenseNet, configuring the final layer for your specific number of classes.
densenet_model = DenseNet(num_classes=num_classes)

# %%
# Use CrossEntropyLoss as loss function
loss_function = nn.CrossEntropyLoss()

# Use Adam with lr=0.001
optimizer = optim.Adam(densenet_model.parameters(), lr=0.001)

