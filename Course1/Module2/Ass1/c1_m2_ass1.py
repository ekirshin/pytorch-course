#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:59:01 2025

@author: ek
"""
import os
import time
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as F

import helper_utils

#%% Terget device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

#%% Load dataset
# Define the path where the EMNIST data will be stored
data_path = './EMNIST_data'

# Check if the data folder exists to avoid re-downloading
if os.path.exists(data_path) and os.path.isdir(data_path):
    download = False
    print("EMNIST Data folder found locally. Loading from local.\n")
else:
    download = True
    print("EMNIST Data folder not found locally. Downloading data.\n")


# Load the EMNIST Letters training set
train_dataset = datasets.EMNIST(
    root=data_path,  # Specify the root directory for the dataset
    split='letters',  # Use the 'letters' subset (26 lowercase classes)
    train=True,  # Indicate that this is the training set
    download=download  # Download the dataset if needed (based on the previous check)
)

test_dataset = datasets.EMNIST(
    root=data_path,  # Specify the root directory for the dataset
    split='letters',  # Use the 'letters' subset (26 lowercase classes)
    train=False,  # Indicate that this is the test set
    download=download  # Download the dataset if needed (based on the previous check)
)

#%% Explore data
index = 83002  # Pick an index from the training set

img, label = train_dataset[index]  # Extract the image and its label

helper_utils.visualize_image(img, label)  # Display the image with its label

# The image is stored as a PIL Image object. You'll convert it to a tensor before training.
print(f"\n Image type: {type(img)}")

#%% Preprocess data
# Precomputed mean and std for EMNIST Letters dataset
mean = (0.1736,)
std = (0.3317,)

# Create a transform that converts images to tensors and normalizes them
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors and scales pixel values to [0, 1]
    transforms.Normalize(mean=mean, std=std)   # Applies normalization using the computed mean and std
])

# Assign the transform to both the training and test datasets
train_dataset.transform = transform
test_dataset.transform = transform

#%% Inspect data as raw tensors
# Set the index of the sample image to view
index = 83002

# Retrieve the transformed image tensor and its label from the training set
img_tensor, label = train_dataset[index]

# Print the image tensor to see the raw values (after transformation)
print(img_tensor)

#%% Transform images for visualization purpose only
def correct_image_orientation(image):
    rotated = F.rotate(image, 90) # Rotate the image 90 degrees clockwise
    flipped = F.vflip(rotated) # Flip the image vertically
    return flipped

# Rotate the image and Reflect it
img_transformed = correct_image_orientation(img_tensor)

# Visualize the transformed image
helper_utils.visualize_image(img_transformed, label)

#%% Load data in batches
def create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64):
    """
    Creates DataLoader objects for the EMNIST training and testing datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): The training dataset.
        test_dataset (torch.utils.data.Dataset): The testing dataset.
        batch_size (int, optional): The batch size for the DataLoaders. Defaults to 64.

    Returns:
        tuple (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
            A tuple containing the training and testing DataLoaders (train_loader, test_loader).
    """

    ### START CODE HERE ###

    # Create a DataLoader for the training dataset
    train_dataloader = DataLoader(
        # Pass in the train_dataset
        dataset=train_dataset,
        # Set batch_size
        batch_size=batch_size,
        # Set shuffle
        shuffle=True
    )  

    # Create a DataLoader for the testing dataset
    test_dataloader = DataLoader(
        # Pass in the test_dataset
        dataset=test_dataset,
        # Set batch_size
        batch_size=batch_size,
        # Set shuffle
        shuffle=False
    )  

    ### END CODE HERE ###

    # Return the created DataLoaders
    return train_dataloader, test_dataloader

#%% Check data loaded with the loaders
train_loader, test_loader = create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64)

print("--- Train Loader --- \n")
helper_utils.display_data_loader_contents(train_loader)
print("\n--- Test Loader --- \n")
helper_utils.display_data_loader_contents(test_loader)

#%% Define NN model
def initialize_emnist_model(num_classes=26):
    """
    Initializes a sequential neural network model for EMNIST classification.

    Args:
        num_classes (int): The number of output classes. Defaults to 26.

    Returns:
        tuple: A tuple containing the model, loss function, and optimizer.
               (model, loss_function, optimizer)
    """

    torch.manual_seed(42)  # Set seed for reproducibility

    ### START CODE HERE ###

    # Define the neural network model using nn.Sequential
    model = nn.Sequential(
        # Flatten the input images
        nn.Flatten(),
        # Middle part should not have more than 5 layers. Only Linear and ReLU layers.
        # Note: Hidden layer units should not exceed 256
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 96), nn.ReLU(),
        nn.Linear(96, 64), nn.ReLU(),
        # Final Layer: SHOULD be Linear Layer
        # Output layer with shape nn.Linear(valid inputs to the layer, num_classes outputs)
        nn.Linear(64, num_classes)
    )

    # Define the loss function (Cross-Entropy Loss for multi-class classification)
    loss_function = nn.CrossEntropyLoss()

    # Define the optimizer (Adaptive Moment Estimation)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ### END CODE HERE ###

    return model, loss_function, optimizer

#%% Examine the model
your_model, loss_func, optimizer = initialize_emnist_model(num_classes=26)

print(f"Your model's architecture:\n\n{your_model}\n")
print(f"Your model's loss function: {loss_func}\n")
print(f"Your model's optimizer:\n\n{optimizer}\n")

#%% Training loop function
def train_epoch(model, loss_function, optimizer, train_loader, device, verbose=True):
    """
    Trains the model for one epoch and calculates the average loss and accuracy.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        loss_function (nn.Module): The loss function used for training.
        optimizer (optim.Optimizer): The optimizer used for updating model parameters.
        train_loader (DataLoader): DataLoader for the training dataset, providing batches of data.
        device (torch.device): The device (CPU or CUDA) where the model and data will be moved.
        verbose (bool, optional): If True, prints the average loss and accuracy for the epoch. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - nn.Module: The trained model after one epoch.
            - float: The average loss for the epoch.
            - float: The accuracy percentage for the epoch.

    Note:
        - The target labels in the `train_loader` (EMNIST letters) are 1-indexed and so the function adjusts them
          to 0-indexed before calculating the loss.
    """

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to training mode
    model.train()
    # Initialize running loss to 0
    running_loss = 0.0
    # Initialize the number of correct predictions to 0
    num_correct_predictions = 0
    # Initialize the total number of predictions to 0
    total_predictions = 0

    # Iterate over the batches of data from the training DataLoader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move the inputs and targets to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
        targets = targets - 1

    ### START CODE HERE ###

        # Zero the gradients of the optimizer.
        optimizer.zero_grad()

        # Pass the inputs through the model to get the outputs.
        outputs = model(inputs)

        # Calculate the loss between the outputs and the targets.
        loss = loss_function(outputs, targets)

        # Perform backpropagation to calculate the gradients.
        loss.backward()

        # Update the model parameters using the optimizer.
        optimizer.step()

        # Loss 
        # Accumulate the loss value to running_loss
        loss_value = loss.item()
        # Add current loss value to the total running loss.
        running_loss += loss_value

        # Accuracy
        # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
        predicted_indices = outputs.argmax(dim=1)

        # Compare predicted indices to actual targets
        correct_predictions = predicted_indices.eq(targets)

        # Sum of correct predictions in the current batch.
        num_correct_in_batch = correct_predictions.sum().item()
        
        # Add correct predictions to the total correct predictions.
        num_correct_predictions += num_correct_in_batch

        # Get the batch size from the targets and add it to total predictions.
        batch_size = targets.size(0)
        total_predictions += batch_size


    # Calculate the average loss for the epoch. 
    # Divide the running loss by the number (len of train loader) of batches.
    average_loss = running_loss/len(train_loader)

    # Calculate the accuracy percentage for the epoch. Multiply correct predictions by 100.
    accuracy_percentage = 100. * num_correct_predictions / total_predictions

    ### END CODE HERE ###

    # Conditionally print based on verbose flag
    if verbose:
        print(
            f"Epoch Loss (Avg): {average_loss:.3f} | Epoch Acc: {accuracy_percentage:.2f}%"
        )

    # Return the trained model and average loss
    return model, average_loss

#%% Test-drive the model to see if it is training as expected
### This will take a few seconds to execute
model, loss_function, optimizer = initialize_emnist_model(num_classes=26)

model_one_train_epoch, _ = train_epoch(model=model, # model
                                       loss_function=loss_function, # loss_function
                                       optimizer=optimizer, # optimizer
                                       train_loader=train_loader, # train_loader
                                       device=DEVICE) # DEVICE

#%% Implemented model evaluation function
def evaluate(model, test_loader, device, verbose=True):
    """
    Evaluates the model on the test dataset and returns the accuracy percentage.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        test_loader (DataLoader): DataLoader for the testing dataset.
        device (torch.device): The device to use (CPU or CUDA).
        verbose (bool, optional): If True, prints the test accuracy. Defaults to True.

    Returns:
        float: The accuracy percentage of the model on the test dataset.

    Note:
        - The target labels in the `test_loader` (EMNIST letters) are 1-indexed and so the function adjusts them 
          to 0-indexed before calculating the accuracy.
    """
    
    # Set the model to evaluate mode
    model.eval()
    # Initialize the number of correct predictions to 0
    num_correct_predictions = 0
    # Initialize the total number of predictions to 0
    total_predictions = 0 

    ### START CODE HERE ###
    
    # Use torch.no_grad() context to disable gradient calculations during evaluation.
    with torch.no_grad():

    ### END CODE HERE ###
        
        # Iterate over the batches of data from the test DataLoader
        for inputs, targets in test_loader:
            # Move the inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
            targets = targets - 1

        ### START CODE HERE ###

            # Pass the inputs through the model to get the outputs.
            outputs = model(inputs)

            # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
            predicted_indices = outputs.argmax(dim=1)

            # Compare predicted indices to actual targets
            correct_predictions = predicted_indices.eq(targets)        
            
            # Sum of correct predictions in the current batch.
            num_correct_in_batch = correct_predictions.sum().item()
            
            # Add correct predictions to the total correct predictions.
            num_correct_predictions += num_correct_in_batch

            # Get the batch size from the targets and add it to total predictions.
            batch_size = targets.size(0)
            total_predictions += batch_size

        ### END CODE HERE ###
        
        # Calculate the accuracy percentage. Multiply correct predictions by 100.
        accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:  # Conditionally print based on verbose flag
        print((f'Test Accuracy: {accuracy_percentage:.2f}%'))

    return accuracy_percentage

#%% Evaluate pre-trained model
model_evaluate = evaluate(model, test_loader, DEVICE)

#%% Complete training and evaluation
def train_and_evaluate(
    model, train_loader, test_loader, num_epochs, loss_function, optimizer, device
):
    """
    Trains and evaluates the model for the specified number of epochs.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        num_epochs (int): The number of epochs to train for.
        loss_function (nn.Module): The loss function used for training.
        optimizer (optim.Optimizer): The optimizer used for updating model parameters.
        device (torch.device): The device to use (CPU or CUDA).

    Returns:
        nn.Module: The trained model.
    """
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}")

        ### START CODE HERE ###
        # Train the model for one epoch using the train_epoch function.
        trained_model, _ = train_epoch(model, loss_function, optimizer, train_loader, device)

        # Evaluate the trained model on the test dataset using the evaluate function.
        accuracy = evaluate(model, test_loader, device)

        ### END CODE HERE ###

    return trained_model

#%% Train the model
# Set the number of training epochs. Don't set more than 15
num_epochs = 10

# Initialize the EMNIST model, loss function, and optimizer
emnist_model, loss_function, optimizer = initialize_emnist_model(num_classes=26)

# --- Benchmark setup ---
loop_start = time.time()

# Train and evaluate the model for the specified number of epochs
trained_model = train_and_evaluate(
    model=emnist_model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=num_epochs,
    loss_function=loss_function,
    optimizer=optimizer,
    device=DEVICE
)

# --- Benchmark results ---
loop_end = time.time()
total_loop_time = loop_end - loop_start

print(f"Number of epochs: {num_epochs}")
print(f"Target device is: {DEVICE}")
print(f"Total loop execution time: {total_loop_time:.3f} seconds")

#%% # Evaluate the trained model's performance on each letter class
class_accuracies = helper_utils.evaluate_per_class(trained_model, test_loader, DEVICE)

# Print the accuracy for each letter class
for letter, accuracy in class_accuracies.items():
    print(f"Accuracy for {letter}: {(accuracy*100):.2f} %")

#%% Decode the "secret" message from Andrew Ng
# get the images of the secret message
message_imgs = helper_utils.load_hidden_message_images()

#%% The decoding function
def decode_word_imgs(word_imgs, model, device):
    # put the model in eval mode
    model.eval()


    decoded_chars = []
    
    with torch.no_grad():
        # iterate through each character image in the word
        for char_img in word_imgs:
            # add batch dimension and move to device
            char_img = char_img.unsqueeze(0).to(device)

            # predict the character with the model
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()

            # match the predicted label to the corresponding character
            lowercase_char = chr(ord("a") + predicted_label)

            # append the character to the decoded_chars list
            decoded_chars.append(f"{lowercase_char}")

    # join the characters to form the decoded word
    decoded_word = "".join(decoded_chars)
    return decoded_word

#%% Decode!
for sentence_imgs in message_imgs:
    decoded_sentence = []

    for word_imgs in sentence_imgs:
        decoded_word = decode_word_imgs(word_imgs, trained_model, DEVICE)
        decoded_sentence.append(decoded_word)

    print(" ".join(decoded_sentence))
    
#%% # WE STRONGLY RECOMMEND YOU TO TRY YOUR OWN ARCHITECTURES FIRST
# AND ONLY RUN THIS CELL IF YOU WISH TO SEE AN ANSWER
import base64

encoded_answer = "LSBBIEZsYXR0ZW4gbGF5ZXIKLSBBIExpbmVhciBsYXllciB3aXRoIDc4NCBpbnB1dCBmZWF0dXJlcyBhbmQgMjU2IG91dHB1dCBmZWF0dXJlcwotIEEgUmVMVSBhY3RpdmF0aW9uIGZ1bmN0aW9uCi0gQSBMaW5lYXIgbGF5ZXIgd2l0aCAyNTYgaW5wdXQgZmVhdHVyZXMgYW5kIDEyOCBvdXRwdXQgZmVhdHVyZXMKLSBBIFJlTFUgYWN0aXZhdGlvbiBmdW5jdGlvbgotIEEgTGluZWFyIGxheWVyIHdpdGggMTI4IGlucHV0IGZlYXR1cmVzIGFuZCBudW1fY2xhc3Nlcz0yNiBvdXRwdXQgZmVhdHVyZXMKLSBUcmFpbiB0aGUgbW9kZWwgZm9yIDEwIG9yIG1vcmUgZXBvY2hz"
encoded_answer = encoded_answer.encode('ascii')
answer = base64.b64decode(encoded_answer)
answer = answer.decode('ascii')

print(answer)