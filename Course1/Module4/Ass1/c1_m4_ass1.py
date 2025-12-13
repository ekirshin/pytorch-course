#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:14:28 2025

@author: ek
"""
import copy 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import helper_utils

#%% # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% Defining More Powerful Transformations

# Pre-calculated mean for each of the 3 channels of the CIFAR-100 dataset
cifar100_mean = (0.5071, 0.4867, 0.4408)
# Pre-calculated standard deviation for each of the 3 channels of the CIFAR-100 dataset
cifar100_std = (0.2675, 0.2565, 0.2761)

#%% # GRADED FUNCTION: define_transformations

def define_transformations(mean, std):
    """
    Creates image transformation pipelines for training and validation.

    Args:
        mean (list or tuple): A sequence of mean values for each channel.
        std (list or tuple): A sequence of standard deviation values for each channel.

    Returns:
        train_transformations (torchvision.transforms.Compose): The training
                                                                transformation pipeline.
        val_transformations (torchvision.transforms.Compose): The validation
                                                                transformation pipeline.
    """
    
    ### START CODE HERE ###
    
    # Define the sequence of transformations for the training dataset.
    
    train_transformations = transforms.Compose([
        # Randomly flip the image horizontally with a 50% probability.
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly flip the image vertically with a 50% probability.
        transforms.RandomVerticalFlip(p=0.5),
        # Rotate the image by a random angle between -15 and +15 degrees.
        transforms.RandomRotation(degrees=15),
        # Convert the image from a PIL Image or NumPy array to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor image with the given mean and standard deviation.
        transforms.Normalize(mean, std)
    ]) 
    
    # Define the sequence of transformations for the validation dataset.
    val_transformations = transforms.Compose([
        # Convert the image from a PIL Image or NumPy array to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor image with the given mean and standard deviation.
        transforms.Normalize(mean, std)
    ]) 
    
    ### END CODE HERE ###

    # Return both transformation pipelines.
    return train_transformations, val_transformations

#%% Verify the Transformations
print("--- Verifying define_transformations ---\n")
train_transform_verify, val_transform_verify = define_transformations(cifar100_mean, cifar100_std)


print("Training Transformations:")
print(train_transform_verify)
print("-" * 30)
print("\nValidation Transformations:")
print(val_transform_verify)

#%% # Create and store the training and validation transformation pipelines
train_transform, val_transform = define_transformations(cifar100_mean, cifar100_std)

#%% Assembling the Data Loaders
# With your powerful new transformation pipelines defined, it is time to prepare the data for training. You will first specify the 15 target classes and then use your transformations to load the images and wrap them in DataLoader objects, which will feed the data to your model in batches.

# First, define the all_target_classes list.
# These are the same classes of flowers, mammals, and insects you worked with in the previous lab, ensuring you are tackling the same classification problem, but with an upgraded pipeline.
# Define the full class list.
all_target_classes = [
    # Flowers
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    # Mammals
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    # Insects
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'
]

#%% Next, call the load_cifar100_subset function, passing in your class list (all_target_classes) and both transformation pipelines (train_transform and val_transform).
# This function handles the entire loading process and returns two PyTorch Dataset objects, which are stored in the train_dataset and val_dataset variables.
# Load the full datasets.
train_dataset, val_dataset = helper_utils.load_cifar100_subset(all_target_classes, train_transform, val_transform)

#%% With your datasets prepared, the final step is to wrap them in PyTorch's DataLoader. This utility is essential for feeding data to your model in manageable batches.
# Create the train_loader for your training data.
# Create the val_loader for your validation data.
# Set the number of samples to be processed in each batch
batch_size = 64

# Create a data loader for the training set, with shuffling enabled
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create a data loader for the validation set, without shuffling
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%% Visualizing the Training Images
# Visualize a grid of random training images
helper_utils.visualise_images(train_loader, grid=(3, 5))

#%% Building a Modular and Robust CNN
# With a more robust data pipeline in place, your next step is to enhance the model's architecture itself. You will refactor the original CNN to be more modular, efficient, and powerful. This is the next pivotal step toward resolving the overfitting problem and pushing your model's performance to new heights.


# 2.1 - The Power of Modularity: The CNNBlock
# In the previous lab, your model's architecture had a repeating pattern of convolution, activation, and pooling layers. Defining these layers individually can become repetitive and makes the model harder to modify. A much better approach is to group these patterns into a single, reusable module. Your first task is to create a CNNBlock that packages these layers together. This modular design makes your main model's code significantly cleaner and easier to manage.


# 2.1.1 - BatchNorm2d Layer
# As part of this new, improved block, you will also introduce a powerful new layer: BatchNorm2d. This layer is a pivotal technique for building modern, high performing deep neural networks.

# Think of Batch Normalization as a traffic controller for the data flowing between your network's layers. After a convolutional layer processes a batch of images, the outputs (or activations) can have widely varying distributions from one batch to the next. BatchNorm2d steps in and normalizes these activations within each mini batch, adjusting them to have a consistent mean and standard deviation. It then uses two learnable parameters to scale and shift this normalized output, allowing the network itself to learn the optimal distribution for the data at that point.

# This seemingly simple step provides three profound benefits:

# It Stabilizes and Accelerates Training: By keeping the distribution of data consistent between layers, it prevents later layers from having to constantly adapt to a shifting input from the layers before them. This stability allows you to use higher learning rates, which can dramatically speed up how quickly your model learns.

# It Acts as a Regularizer: Because the normalization statistics are calculated for each unique mini batch, it introduces a slight amount of noise into the training process. This noise makes it harder for the model to perfectly memorize the training data, encouraging it to learn more general features and thus reducing overfitting.

# It Reduces Sensitivity to Initialization: The layer makes your model less dependent on the specific random weights it starts with, leading to more reliable and repeatable training results.

# By adding BatchNorm2d to your CNNBlock, you are not just adding another layer; you are fundamentally making your model's training process more stable, efficient, and robust.

class CNNBlock(nn.Module):
    """
    Defines a single convolutional block for a CNN.

    This block consists of a convolutional layer, batch normalization,
    a ReLU activation, and a max-pooling layer, bundled as a sequential module.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Initializes the layers of the CNNBlock.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        """
        # Initialize the parent nn.Module class.
        super(CNNBlock, self).__init__()
        
        ### START CODE HERE ###
        
        # Define the sequential container for the block's layers.
        self.block = nn.Sequential(
            # 2D convolutional layer to apply learnable filters to the input.
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            # Batch normalization to stabilize and accelerate training.
            nn.BatchNorm2d(out_channels),
            # ReLU activation function to introduce non-linearity.
            nn.ReLU(),
            # Max pooling layer to downsample the feature map and reduce spatial dimensions.
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        
        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass for the CNNBlock.

        Args:
            x: The input tensor for the block.

        Returns:
            The output tensor after passing through the block.
        """
        
        ### START CODE HERE ###
        
        # Pass the input tensor through the sequential block of layers.
        return self.block(x)
    
        ### END CODE HERE ###

#%% # Verify the CNNBlock
print("--- Verifying CNNBlock ---\n")

# Instantiate the block with 3 input channels and 16 output channels
verify_cnn_block = CNNBlock(in_channels=3, out_channels=16)
print("Block Structure:\n")
print(verify_cnn_block)

# Verify the output shape after a forward pass
# Create a dummy input tensor (batch_size=1, channels=3, height=32, width=32)
dummy_input = torch.randn(1, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

# Pass the dummy tensor through the block
output = verify_cnn_block(dummy_input)
print(f"Output tensor shape: {output.shape}")

#%% 2.2 - Assembling the Full CNN with Modular Blocks
# Now that you have a reusable CNNBlock, you can assemble your full SimpleCNN architecture. By using your new modular block, you will see how much cleaner and more professional your model definition becomes. Instead of defining many individual layers for the convolutional part of your network, you will now define just three CNNBlock instances.

# Your model will consist of two main parts:

# A feature extractor: A sequence of three CNNBlocks that will learn to identify visual patterns in the images.
# A classifier: A sequence of fully connected layers that will take the features from the convolutional blocks and make the final prediction.
# In this new version, you will also increase the dropout rate to 0.6. This is another important step in your fight against overfitting, as it makes the model less likely to rely on any single feature.
class SimpleCNN(nn.Module):
    """
    Defines a simple CNN architecture using modular CNNBlocks.

    This model stacks three reusable convolutional blocks followed by a fully
    connected classifier to perform image classification.
    """
    def __init__(self, num_classes):
        """
        Initializes the layers of the SimpleCNN model.

        Args:
            num_classes (int): The number of output classes for the classifier.
        """
        # Initialize the parent nn.Module class.
        super(SimpleCNN, self).__init__()
        
        ### START CODE HERE ###

        # Define the first convolutional block.
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        # Define the second convolutional block.
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        # Define the third convolutional block.
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # Define the fully connected classifier block.
        self.classifier = nn.Sequential(
            # Flatten the 3D feature map (channels, height, width) into a 1D vector.
            nn.Flatten(),
            # First fully connected (linear) layer that maps the flattened features to a hidden layer.
            nn.Linear(128*4*4, 512),
            # ReLU activation function to introduce non-linearity.
            nn.ReLU(),
            # Dropout layer to prevent overfitting by randomly setting a fraction of inputs to zero.
            nn.Dropout(p=0.6),
            # Final fully connected (linear) layer that maps the hidden layer to the output classes.
            nn.Linear(512, num_classes)
        ) 
        
        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): The input tensor containing a batch of images.

        Returns:
            torch.Tensor: The output tensor with logits for each class.
        """
        
        ### START CODE HERE ###
        
        # Pass the input through the first convolutional block.
        x = self.conv_block1(x)
        # Pass the result through the second convolutional block.
        x = self.conv_block2(x)
        # Pass the result through the third convolutional block.
        x = self.conv_block3(x)

        # Pass the final feature map through the classifier.
        x = self.classifier(x)
        
        ### END CODE HERE ###
        
        # Return the final output tensor.
        return x

#%% Verify the SimpleCNN
print("--- Verifying SimpleCNN ---\n")

# Verify the structure of the model
# Instantiate the model with 15 output classes
verify_simple_cnn = SimpleCNN(num_classes=15)
print("Model Structure:\n")
print(verify_simple_cnn)

# Verify the output shape after a forward pass
# Create a dummy input tensor (batch_size=64, channels=3, height=32, width=32)
dummy_input = torch.randn(64, 3, 32, 32)
print(f"\nInput tensor shape:  {dummy_input.shape}")

# Pass the dummy tensor through the model
output = verify_simple_cnn(dummy_input)
print(f"Output tensor shape: {output.shape}")

#%% With your SimpleCNN class defined, the next step is to create an instance of the model.

# First, dynamically determine the number of classes by getting the length of the .classes attribute from your train_dataset.
# Next, create an instance of your SimpleCNN model, passing the num_classes variable to its constructor. This ensures the final layer of your model is correctly sized for your 15-class problem.
# Get the number of classes
num_classes = len(train_dataset.classes)

# Instantiate the model
model = SimpleCNN(num_classes)

#%% Training the Upgraded Model
# With your upgraded data pipeline and modular CNN architecture complete, you are ready to begin the training process. In this section, you will configure the final pieces of your training pipeline: the loss function and the optimizer. Then, you will implement the core training and validation logic that will run your experiment and reveal how well your new model performs.
# 3.1 - Configuring the Loss and Optimizer
# Before you can train the model, you must define two key components: a loss function to measure error and an optimizer to update the model's weights.

# For the loss function, you will continue to use nn.CrossEntropyLoss, the standard choice for multi-class classification.
# For the optimizer, you will use Adam, but with an important addition to combat overfitting: weight_decay.
# Weight decay adds a penalty to the loss function based on the magnitude of the model's weights. It encourages the network to learn smaller, simpler weight values, which makes it more robust and less likely to memorize the training data. This is another vital tool for improving your model's ability to generalize.
# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer for the model with weight_decay
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

#%%Implementing the Training and Validation Logic
# You will now implement the core logic for training and evaluating your model. This will be done in two separate functions:

# train_epoch: To perform a single pass over the training data to update the model.
# validate_epoch: To perform a single pass over the validation data to measure performance.

# Exercise 4 - train_epoch
# Your task is to complete the core training logic within the for loop of the train_epoch function. You will implement the five fundamental steps of a single training iteration.

# Your Task:

# Inside the train_epoch function, for each batch of images and labels:

# Clear Gradients:
# Before computing the gradients for the current batch, you must clear any gradients that were stored from the previous batch.
# Forward Pass:
# Feed the images through the model to get the output predictions.
# Calculate Loss:
# Use the provided loss_function to measure the difference between the model's outputs and the true labels.
# Backward Pass:
# Compute the gradients of the loss with respect to all the model's parameters. This is also known as backpropagation.
# Update Parameters:
# Use the optimizer to adjust the model's parameters based on the gradients you just computed.
def train_epoch(model, train_loader, loss_function, optimizer, device):
    """
    Performs a single training epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        float: The average training loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    # Iterate over batches of data in the training loader
    for images, labels in train_loader:
        # Move images and labels to the specified device
        images, labels = images.to(device), labels.to(device)
        
        ### START CODE HERE ###
        
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Perform a forward pass to get model outputs
        outputs = model(images)
        # Calculate the loss
        loss = loss_function(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        
        ### END CODE HERE ###
        
        # Accumulate the training loss for the batch
        running_loss += loss.item() * images.size(0)
        
    # Calculate and return the average training loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

#%% # Use a helper function to perform a sanity check on the train_epoch implementation
helper_utils.verify_training_process(SimpleCNN, train_loader, loss_function, train_epoch, device)

#%% Exercise 5 - validate_epoch¶
# Your task is to complete the validation logic. This involves performing a forward pass and then calculating both the loss and the number of correct predictions to determine the accuracy.

# Your Task:

# Disable Gradient Calculation:
# Wrap the entire for loop within the torch.no_grad() context manager. This tells PyTorch not to compute gradients, which saves memory and computation time during validation.
# Inside the for loop:
# Forward Pass:
# Just like in training, pass the images through the model to get its outputs.
# Calculate Loss:
# Use the loss_function to compute the val_loss between the outputs and the true labels.
# Accumulate Loss:
# Add the batch's loss to the running_val_loss. Remember to get the scalar value from the loss tensor and scale it by the batch size.
# Get Predictions:
# Determine the model's predicted class for each image in the batch. The outputs from your model are raw scores (logits). The class with the highest score is the model's prediction. You need to find the index of this maximum score.

def validate_epoch(model, val_loader, loss_function, device):
    """
    Performs a single validation epoch.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        loss_function (callable): The loss function.
        device (torch.device): The device (CPU or GPU) to perform validation on.

    Returns:
        tuple: A tuple containing the average validation loss and validation accuracy.
    """
    # Set the model to evaluation mode
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    
    ### START CODE HERE ###
    
    # Disable gradient calculations for validation
    with torch.no_grad():
        
    ### END CODE HERE ###
    
        # Iterate over batches of data in the validation loader
        for images, labels in val_loader:
            # Move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)
            
            ### START CODE HERE ###
            
            # Perform a forward pass to get model outputs
            outputs = model(images)
            
            # Calculate the validation loss for the batch
            val_loss = loss_function(outputs, labels).item()
            # Accumulate the validation loss
            running_val_loss += val_loss * images.size(0)
            
            # Get the predicted class labels
            _, predicted = outputs.max(dim=1)
            
            ### END CODE HERE ###
            
            # Update the total number of samples
            total += labels.size(0)
            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()
            
    # Calculate the average validation loss and accuracy for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_val_loss, epoch_accuracy

#%% # Use a helper function to perform a sanity check on the validate_epoch implementation
helper_utils.verify_validation_process(SimpleCNN, val_loader, loss_function, validate_epoch, device)

#%% Training
# With the individual functions for training and validation complete, you can now bring them together in the main training_loop. This function orchestrates the entire training process over a set number of epochs and includes a pivotal upgrade.

# A common challenge is that a model's performance can peak and then decline if training continues for too long. To address this, the training_loop will:

# Monitor the validation accuracy at the end of each epoch.
# Keep track of the best performing model state seen so far.
# After the final epoch, it automatically returns the model from its single best epoch.
# This guarantees that you always get back the version of your model that achieved the highest validation accuracy during the entire training run.
def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run training on.

    Returns:
        tuple: A tuple containing the best trained model and a list of metrics
               (train_losses, val_losses, val_accuracies).
    """
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    # Initialize variables to track the best performing model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Initialize lists to store training and validation metrics
    train_losses, val_losses, val_accuracies = [], [], []
    
    print("--- Training Started ---")
    
    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Perform one epoch of training
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)
        
        # Perform one epoch of validation
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())
            
    print("--- Finished Training ---")
    
    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    
    # Return the trained model and the collected metrics
    return model, metrics

#%% 
# Start the training process by calling the training loop function
trained_model, training_metrics = training_loop(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    loss_function=loss_function, 
    optimizer=optimizer, 
    num_epochs=50, 
    device=device
)

# Visualize the training metrics (loss and accuracy)
print("\n--- Training Plots ---\n")
helper_utils.plot_training_metrics(training_metrics)