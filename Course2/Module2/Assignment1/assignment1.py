# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:48:36 2026

@author: ekirshin
"""

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import helper_utils

#%% 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

#%% 
# Load the dataset path
dataset_path = "./AIvsReal_sampled"  

# Analyzes the dataset splits at the given path and prints a count of images for each class.
helper_utils.dataset_images_per_class(dataset_path)

#%% 
# Randomly select and display a grid of sample images from the 'train' folder.
helper_utils.display_train_images(dataset_path)

#%% Exercise 1 - create_dataset_splits
# GRADED FUNCTION: create_dataset_splits

def create_dataset_splits(data_path):
    """
    Creates training and validation datasets from a directory structure using ImageFolder.

    Args:
        data_path (str): The root path to the dataset directory, which should
                         contain 'train' and 'validation/test' subdirectories.

    Returns:
        tuple: A tuple containing the train_dataset and validation_dataset
               (train_dataset, validation_dataset).
    """

    # Construct the full path to the training data directory.
    train_path = data_path + "/train"
    # Construct the full path to the validation data directory.
    val_path = data_path + "/test"

    ### START CODE HERE ###
    
    # Create the train dataset using ImageFolder
    train_dataset = ImageFolder(
        # Set the root to train dataset path
        root=train_path,
    ) 

    # Create the validation dataset using ImageFolder
    val_dataset = ImageFolder(
        # Set the root to validation dataset path
        root=val_path,
    ) 

    ### END CODE HERE ###

    return train_dataset, val_dataset

#%% # Verify that the function loads the datasets
temp_train, temp_val = create_dataset_splits(dataset_path)

print("--- Training Dataset ---")
print(temp_train)
print("\n--- Validation Dataset ---")
print(temp_val)

#%% 1.2 - Specialized Augmentations: Protocols for Training and Validation
# Define the standard mean values for the ImageNet dataset
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])

# Define the standard standard deviation values for the ImageNet dataset
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

#%% 
# GRADED FUNCTION: define_transformations

def define_transformations(mean=imagenet_mean, std=imagenet_std):
    """
    Defines separate series of image transformations for training and validation datasets.

    Args:
        mean (list or tuple): The mean values (for each channel, e.g., RGB) calculated from ImageNet.
        std (list or tuple): The standard deviation values (for each channel) calculated from ImageNet.

    Returns:
        tuple: A tuple containing two `torchvision.transforms.Compose` objects:
               - The first for training transformations.
               - The second for validation transformations.
    """

    ### START CODE HERE ###

    # Create a Compose object to chain multiple transformations together for the training set
    
    # Initialize 'train_transform' using transforms.Compose to apply a sequence of transforms
    train_transform = transforms.Compose([
        # Randomly resize and crop the input image to 224x224 pixels
        transforms.RandomResizedCrop(size=(224, 224)),

        # Apply a random horizontal flip to the image for data augmentation
        transforms.RandomHorizontalFlip(p=0.5),

        # Randomly change the brightness and contrast of the image for data augmentation
        # Set `brightness=0.2` and `contrast=0.2`
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        # Convert the PIL Image to a PyTorch Tensor
        transforms.ToTensor(),

        # Normalize the tensor image with the provided 'mean' and 'std' to normalize the tensor
        transforms.Normalize(mean=mean, std=std),
    ]) 

    # Create a Compose object to chain multiple transformations together for the validation set
    
    # Initialize 'val_transform' using transforms.Compose to apply a sequence of transforms
    val_transform = transforms.Compose([
        # Resize the input image to to 224x224 pixels
        transforms.Resize(size=(224, 224)),

        # Convert the PIL Image to a PyTorch Tensor
        transforms.ToTensor(),

        # Normalize the tensor image with the provided 'mean' and 'std' to normalize the tensor
        transforms.Normalize(mean=mean, std=std),
    ]) 

    ### END CODE HERE ###

    return train_transform, val_transform

#%% # Create the composed transformations
combined_transformations = define_transformations()

# Print the composed transformations to verify the sequence of operations
print("Augmented Training Transformations:\n")
print(combined_transformations[0])
print("\nValidation Transformations:\n")
print(combined_transformations[1])

#%% 1.3 - Preparing the Data Launchpad: Applying Transformations and Batching Data

# GRADED FUNCTION: create_data_loaders

def create_data_loaders(trainset, valset, batch_size):
    """
    Creates DataLoader instances for training and validation datasets with respective transformations.

    Args:
        trainset (torch.utils.data.Dataset): The training dataset.
        valset (torch.utils.data.Dataset): The validation dataset.
        batch_size (int): The number of samples to load in each batch.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - trainset (torch.utils.data.Dataset): The original training dataset with transformations now applied.
            - valset (torch.utils.data.Dataset): The original validation dataset with transformations now applied.
    """
    
    ### START CODE HERE ###
    
    # Define separate transformations for the training and validation datasets
    # Use define_transformations() to get train_transform and val_transform
    train_transform, val_transform = define_transformations()
    
    # Apply the train transformations directly to the train dataset by setting the .transform attribute
    trainset.transform = train_transform
    # Apply the val transformations directly to the val dataset by setting the .transform attribute
    valset.transform = val_transform
    
    # Create a DataLoader for the training dataset
    # Use the transformed train dataset
    # Set batch_size to the input batch_size
    # Set shuffle=True
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Create a DataLoader for the validation dataset
    # Use the transformed validation dataset
    # Set batch_size to the input batch_size
    # Set shuffle=False
    val_loader  = DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    ### END CODE HERE ###
    
    return train_loader, val_loader, trainset, valset

#%% 
dataloaders = create_data_loaders(temp_train, temp_val, batch_size=16)

print("--- Train Loader ---")
helper_utils.display_data_loader_contents(dataloaders[0])
print("\n--- Val Loader ---")
helper_utils.display_data_loader_contents(dataloaders[1])

#%% 2 - Advanced Expedition: Leveraging Pre-trained Model Wisdom
# GRADED FUNCTION: load_mobilenetv3_model

def load_mobilenetv3_model(weights_path):
    """
    Loads a pre-trained MobileNetV3-Large model from torchvision.

    Args:
        weights_path (str): The file path to the saved .pth model weights.
        
    Returns:
        torch.nn.Module: A pre-trained MobileNetV3-Large model.
    """
    ### START CODE HERE ###

    # Load the pre-trained MobileNetV3-Large model without pre-trained weights.
    model = tv_models.mobilenet_v3_large(weights=None)

    # Load the state dictionary (weights) from the local file.
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    ### END CODE HERE ###
    
    model.load_state_dict(state_dict)

    return model

#%% # Load the pre-trained MobileNetV3-Large model using weights from the local file.
local_weights = "./mobilenet_weights/mobilenet_v3_large-8738ca79.pth"
test_model = load_mobilenetv3_model(local_weights)

# Print the last layer of the classifier of the loaded model
print(test_model.classifier[-1])

#%%  Tailoring Your Champion: Adapting MobileNetV3 for Fake Detection
# GRADED FUNCTION: update_model_last_layer

def update_model_last_layer(model, num_classes):
    """
    Freezes the feature layers of a pre-trained model and replaces its final
    classification layer with a new one adapted to the specified number of classes.

    Args:
        model (torch.nn.Module): The pre-trained model to be modified.
        num_classes (int): The number of output classes for the new classification layer.

    Returns:
        torch.nn.Module: The modified model with frozen feature layers and a new
                         classification layer.
    """
    ### START CODE HERE ###

    # Freeze the parameters of the feature layers of the model
    # Iterate through each parameter in model.features.parameters()
    for feature_parameter in model.features.parameters():
        # Set the requires_grad attribute of each feature_parameter to False
        feature_parameter.requires_grad = False

    # Access the final classification layer of the model
    last_classifier_layer = model.classifier[-1] 
    
    # Access the in_features attribute of last_classifier_layer
    num_features = last_classifier_layer.in_features
    
    # # Use nn.Linear to create a new Linear layer for classification with the original number of
    # input features and the specified number of output classes
    new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)
    
    # Replace the original last classification layer with the newly created layer
    model.classifier[-1] = new_classifier
    
    ### END CODE HERE ###

    return model

#%% # Modify the last layer of the MobileNetV3-Large model
test_model = update_model_last_layer(test_model, num_classes=5)

# Print the last layer of the classifier of the modified model
print(test_model.classifier[-1]) 

#%% Final Expedition: Training and Evaluating Your Adapted Champion
# Before you can begin the actual training, you first need to set up your environment by executing three core steps:

# Initializing the Datasets: You'll first load your training and validation data from their respective directories.

# Initializing the DataLoaders: Next, you'll create the train_loader and val_loader to feed your transformed images to the model in manageable batches.

# Preparing the Model: You'll load the pre-trained MobileNetV3-Large model and then adapt its final classification layer for your two specific classes. This readies your champion model for its specialized mission.

# The following code cell will execute these preparatory steps, ensuring both your data and your model are primed for the training loop.
# Initialize the training and validation datasets
train_dataset, val_dataset = create_dataset_splits(dataset_path)

# Initialize the dataloaders for training and validation
train_loader, val_loader, _, __ = create_data_loaders(train_dataset, val_dataset, batch_size=32)

#%% 
# Load the pre-trained MobileNetV3-Large model and modify its last layer
local_weights = "./mobilenet_weights/mobilenet_v3_large-8738ca79.pth"
mobilenet_model = load_mobilenetv3_model(local_weights)
mobilenet_model = update_model_last_layer(mobilenet_model, num_classes=2)

#%% Training the Adapted Model
# Define the loss function to compute the difference between the model's output and the true labels
loss_fcn = nn.CrossEntropyLoss()

# Define the optimizer to update the model's weights during training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                             mobilenet_model.parameters()), 
                       lr=0.001)

#%% Train the model
# Set the number of epochs
num_epochs = 1
# Train the model
trained_model = helper_utils.training_loop(
    mobilenet_model, 
    train_loader, 
    val_loader,
    loss_fcn,
    optimizer,
    DEVICE, 
    num_epochs
)

#%% Performance Review: Visualizing Predictions
# Get the list of class names ('fake', 'real') from the validation dataset.
class_names = val_dataset.classes

# Visualise predictions made by the trained model
helper_utils.visualize_predictions(trained_model, val_loader, DEVICE, class_names)

#%% Test on random images
image_path = './images/fake/birds_sheep_dog.jpg' ### Add your image path here

# Display a prediction for the single uploaded image.
helper_utils.make_predictions(trained_model, image_path, DEVICE, class_names)
