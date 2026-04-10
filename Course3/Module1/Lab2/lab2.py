# -*- coding: utf-8 -*-
"""
C3 M1 L2
Created on Fri Mar 27 11:49:43 2026

@author: ekirshin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torchvision import transforms

import helper_utils

# Set seed
SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Set the path to the root directory of the image dataset.
dataset_path = "./Aerial_Landscapes/"

# %%
# Display the image count statistics for each class.
helper_utils.display_dataset_stats(dataset_path)

# %% Assemble Your Data Pipeline
# Pre-calculated mean and std of this dataset
mean = [0.378, 0.393, 0.345]
std = [0.205, 0.173, 0.170]

# Transformations for the training set (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(100, 100), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Transformations for validation set (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])

# %%
# Create the training and validation datasets by splitting the main dataset.
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
# %%
# Define the number of images to process in each batch.
batch_size = 32

# Create the training and validation DataLoaders using the helper function.
train_loader, val_loader = helper_utils.create_dataloaders(train_dataset, val_dataset, batch_size)

# %% # Display the sample images from train set
helper_utils.show_sample_images(train_dataset)

# %% Architecting Your Neural Networks
# The PlainCNN: A Baseline for Comparison
class PlainBlock(nn.Module):
    """
    A basic two-layer convolutional block without skip connections.

    Args:
        in_channels (int): The number of channels in the input feature map.
        out_channels (int): The number of channels produced by the convolutions.
        stride (int, optional): The stride for the first convolutional layer,
                                used for downsampling. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        # Initialize the parent nn.Module.
        super(PlainBlock, self).__init__()

        # First convolutional layer, which handles input channels and potential downsampling.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Defines the forward pass for the PlainBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after two convolutions.
        """
        # Apply the first convolution, batch normalization, and ReLU activation.
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Apply the second convolution and batch normalization.
        # Note: A final activation is typically applied after this block in the main network.
        out = self.bn2(self.conv2(out))
        
        # Return the output feature map.
        return out

# %%
class PlainCNN(nn.Module):
    """
    A plain Convolutional Neural Network for image classification.

    This network is constructed from a series of basic convolutional blocks
    (PlainBlock) without using residual (skip) connections. It features an
    initial convolution, followed by several layers of stacked blocks that
    progressively increase channel depth and reduce spatial dimensions, and
    concludes with a classification head.

    Args:
        num_classes (int, optional): The number of output classes for the final
                                     classification layer. Defaults to 5.
        num_blocks (list of int, optional): A list defining the number of
                                            PlainBlocks in each of the three
                                            main layers. Defaults to [2, 2, 2].
    """
    def __init__(self, num_classes=5, num_blocks=[2, 2, 2]):
        # Initialize the parent nn.Module.
        super(PlainCNN, self).__init__()

        # Initialize the number of input channels for the first main layer.
        self.in_channels = 32
        
        # Initial convolutional block to process the input image.
        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Define the main layers of the network by stacking PlainBlocks.
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)

        # Final block for global average pooling and classification.
        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Builds a layer by stacking multiple PlainBlocks.

        Args:
            out_channels (int): The number of output channels for the blocks.
            num_blocks (int): The number of blocks to stack in this layer.
            stride (int): The stride for the first block, used for downsampling.

        Returns:
            nn.Sequential: A sequential container of the stacked blocks.
        """
        # Initialize an empty list to hold the blocks for this layer.
        layers = []
        
        # The first block in a layer handles downsampling and channel changes.
        layers.append(PlainBlock(self.in_channels, out_channels, stride))
        
        # Update the number of input channels for the subsequent layer.
        self.in_channels = out_channels
        
        # Add the remaining blocks for this layer.
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(self.in_channels, out_channels))
            
        # Return the blocks wrapped in a sequential container.
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass for the PlainCNN.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output logits from the final classification layer.
        """
        # Pass the input through the initial block.
        out = self.initial_block(x)
        
        # Pass through the main layers, each followed by a ReLU activation.
        out = F.relu(self.layer1(out))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        
        # Pass through the final classification block.
        out = self.final_block(out)
        
        # Return the final output logits.
        return out

# %% The SimpleResNet: An Elegant Solution
class ResidualBlock(nn.Module):
    """
    A fundamental building block for ResNet architectures.

    This block implements a residual connection, allowing the network to learn
    an identity function if needed. It consists of a main path with two
    convolutional layers and a "skip connection" that adds the input of the
    block to its output. This helps mitigate the vanishing gradient problem in
    very deep networks.

    Args:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The number of channels produced by the convolutions.
        stride (int, optional): The stride for the first convolutional layer,
                                used for downsampling. Defaults to 1.
        downsample (nn.Module, optional): A module to downsample the input
                                          (identity) so its dimensions match the
                                          output for the skip connection.
                                          Defaults to None.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # Initialize the parent nn.Module.
        super(ResidualBlock, self).__init__()

        # First component of the main path: Conv -> BatchNorm -> ReLU.
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Second component of the main path: Conv -> BatchNorm.
        # Note: The final ReLU is applied after the skip connection.
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Optional downsampling layer for the skip connection.
        self.downsample = downsample

    def _initial_forward(self, x):
        """Defines the main convolutional path of the block."""
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        return out

    def forward(self, x):
        """
        Defines the forward pass, combining the main path and the skip connection.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the residual block.
        """
        # Store the input for the skip connection.
        identity = x

        # Pass the input through the main convolutional path.
        out = self._initial_forward(x)

        # If needed, apply downsampling to the identity to match dimensions.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the original input (identity) to the output of the main path.
        out += identity

        # Apply the final activation function.
        out = F.relu(out)

        # Return the final output of the block.
        return out

# %% Assembling the SimpleResNet Architecture
class SimpleResNet(nn.Module):
    """
    A simplified ResNet-style architecture for image classification.

    This network is built by stacking ResidualBlock modules. It consists of an
    initial convolutional layer, followed by three main stages of residual
    blocks, and a final classification head with global average pooling.

    Args:
        num_classes (int, optional): The number of output classes for the final
                                     classification layer. Defaults to 5.
        num_blocks (list of int, optional): A list defining the number of
                                            ResidualBlocks in each of the three
                                            main stages. Defaults to [2, 2, 2].
    """
    def __init__(self, num_classes=5, num_blocks=[2, 2, 2]):
        # Initialize the parent nn.Module.
        super(SimpleResNet, self).__init__()

        # Store the number of classes for the final layer.
        self.num_classes = num_classes
        
        # Initialize the number of input channels for the first residual stage.
        self.in_channels = 32

        # Define the initial convolutional layer.
        self.initial_block = self._get_initial_block()

        # Construct the main stages of the network using residual blocks.
        self.res_block1 = self._make_residual_block(32, num_blocks[0], stride=1)
        self.res_block2 = self._make_residual_block(64, num_blocks[1], stride=2)
        self.res_block3 = self._make_residual_block(128, num_blocks[2], stride=2)

        # Define the final classification head.
        self.final_block = self._get_final_block()

    def _make_residual_block(self, out_channels, num_blocks, stride):
        """
        Builds a residual stage by stacking multiple ResidualBlocks.

        Args:
            out_channels (int): The number of output channels for the blocks.
            num_blocks (int): The number of blocks to stack in this stage.
            stride (int): The stride for the first block, used for downsampling.

        Returns:
            nn.Sequential: A sequential container of the stacked blocks.
        """
        # Initialize the downsample layer as None.
        downsample = None
        
        # Define the downsample layer if stride is not 1 or if channel dimensions change.
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # Initialize a list to hold the layers for this stage.
        layers = []
        
        # The first block in a stage handles downsampling and channel changes.
        first_block = ResidualBlock(self.in_channels, out_channels, stride, downsample)
        layers.append(first_block)
        
        # Update the number of input channels for the subsequent stage.
        self.in_channels = out_channels

        # Add the remaining residual blocks for this stage.
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        # Return the blocks wrapped in a sequential container.
        return nn.Sequential(*layers)

    def _get_initial_block(self):
        """Constructs the initial convolutional layer of the network."""
        initial_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        return initial_block

    def _get_final_block(self):
        """Constructs the final classification head of the network."""
        final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.num_classes),
        )
        return final_block

    def forward(self, x):
        """
        Defines the forward pass for the SimpleResNet.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output logits from the final classification layer.
        """
        # Pass input through the initial convolutional block.
        out = self.initial_block(x)
        
        # Pass through the three main stages of residual blocks.
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        
        # Pass through the final classification head.
        out = self.final_block(out)
        
        # Return the final output logits.
        return out

# %% Initialize the Models
# Set the random seed to ensure that both models are always initialized with the same random weights.
torch.manual_seed(SEED)

# Create an instance of the SimpleResNet model.
resnet_model = SimpleResNet(num_classes=num_classes)

# Create an instance of the PlainCNN model for your baseline comparison.
plain_model = PlainCNN(num_classes=num_classes)

# %% Examining the Model Architecture
# Define a configuration dictionary to store parameters for the model summary.
config = {
    "input_size": (batch_size, 3, 64, 64),
    "attr_names": ["input_size", "output_size", "num_params", "trainable"],
    "col_names_display": ["Input Shape ", "Output Shape", "Param #", "Trainable"],
    "depth": 3
}

# Generate the model summary object using torchinfo with the specified configuration.
summary = torchinfo.summary(
    model=resnet_model, 
    input_size=config["input_size"], 
    col_names=config["attr_names"], 
    depth=config["depth"]
)

# Display the summary as a styled HTML table.
print("--- Model Summary ---\n")
helper_utils.display_torch_summary(summary, config["attr_names"], config["col_names_display"], config["depth"], save_path="model_summary.html")

# %% Loss Function and Optimizer
# Use CrossEntropyLoss, a standard loss function for multi-class classification tasks.
loss_function = nn.CrossEntropyLoss()

# Create an Adam optimizer to update the weights of the ResNet model.
optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=0.001)

# Create a separate Adam optimizer for the Plain CNN model to train it independently.
optimizer_plain = optim.Adam(plain_model.parameters(), lr=0.001)

# %% Training and Validation
# You will now train both of your models using the provided training_loop_16_mixed training function. This custom loop handles the complete training and validation cycle, and it uses a 16 bit mixed precision strategy to accelerate performance. This optimization is implemented using PyTorch's own built in libraries.
# To judge the performance of the models fairly, the function will track and return the following key results for later comparison:
# history: A Python dictionary that records the performance after each epoch. It contains the training loss, validation loss, and validation accuracy, which are essential for plotting the learning curves to visualize the training progress.
# final_cm: A confusion matrix calculated on the validation data after the final epoch. This is a powerful tool that shows exactly which classes the model is confusing with others, helping you analyze its specific strengths and weaknesses.

# %% # Define the total number of full training cycles (epochs) to run.
num_epochs = 10

# %%
# Training of `plain_model`
trained_plain, history_plain, cm_plain = helper_utils.training_loop_16_mixed(
    model=plain_model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer_plain,
    num_epochs=num_epochs,
    device=device
)

# %%
# With the baseline set, it's time for your advanced SimpleResNet. Run it through the same training process and see if the skip connections give it an edge.
# Training of `resnet_model`
trained_resnet, history_resnet, cm_resnet = helper_utils.training_loop_16_mixed(
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer_resnet,
    num_epochs=num_epochs,
    device=device
)

# %% The Tale of the Tape: Visualizing Performance
# With both models trained, it's time to see which one performed better. You'll now plot the learning curves from both history objects of the trained models.
# This will give you a direct, visual comparison of how the PlainCNN (your baseline) and the SimpleResNet (your advanced model) performed on key metrics like loss and accuracy throughout the training process.
# Plot the training metrics from both models for a direct visual comparison.
helper_utils.plot_training_logs(history_plain, history_resnet)

# %% Visualizing Predictions and Confusion Matrices
# Get the list of class names (e.g., 'Forest', 'River') from the dataset for plotting.
class_names = val_loader.dataset.classes

# Visualize predictions from the SimpleResNet model on the validation data.
helper_utils.visualize_predictions(trained_resnet, val_loader, class_names, device)

# %%
### Uncomment and execute the line below if you wish to see the predictions by the trained PlainCNN model.

# Visualize predictions from the PlainCNN model on the validation data.
helper_utils.visualize_predictions(trained_plain, val_loader, class_names, device)

# %%
# Plot the confusion matrix for the baseline PlainCNN to analyze its error patterns.
helper_utils.plot_confusion_matrix(cm_plain, class_names)

# %%
# Plot the confusion matrix for the ResNet style model to analyze its error patterns.
helper_utils.plot_confusion_matrix(cm_resnet, class_names)


