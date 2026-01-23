# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:08:15 2026

@author: ekirshin
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

import helper_utils

#%% 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

#%% # Define the transformation pipeline
emnist_transformation = transforms.Compose([
    # Convert grayscale image to 3 channels to match MobileNetV2's input
    transforms.Grayscale(num_output_channels=3),
    # Resize the image to 224x224, the standard input size for MobileNetV2
    transforms.Resize((224, 224)),
    # Apply the 90-degree rotation augmentation
    transforms.RandomRotation(degrees=(90, 90)),
    # Apply the vertical flip augmentation
    transforms.RandomVerticalFlip(p=1.0),
    # Convert the image to a PyTorch Tensor
    transforms.ToTensor(),
    # Normalize the tensor using ImageNet's mean and standard deviation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#%% # Create the training and validation DataLoaders.
train_loader, val_loader = helper_utils.create_emnist_dataloaders(
    batch_size=32,
    transform=emnist_transformation  # Apply the defined transformations
)

#%% # Load the pre-trained MobileNetV3 model and set it to evaluation mode for inference
mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()

#%% # Load the mapping of class indices to human-readable names from the JSON file
class_names = helper_utils.load_imagenet_classes('./imagenet_class_index.json')

#%% # Visualize the model's predictions on the validation images
helper_utils.show_predictions(mobilenet_model, val_loader, device, class_names)

#%% Strategy 1: Feature Extraction (Train Only the New Head)
# Direct Attribute
# Instantiate the ResNet18 model architecture and load the selected weights
resnet18_model = tv_models.resnet18(weights='IMAGENET1K_V1')

print(resnet18_model)

#%% Freeze parameters 
# Iterate over each parameter in the resnet18_model
for param in resnet18_model.parameters():
    # Set the requires_grad attribute of each parameter to False to freeze it
    param.requires_grad = False

#%% # Access the fully connected layer (fc) of the resnet18_model
original_fc_layer = resnet18_model.fc

print("Model's Original Fully Connected Layer:")
print(original_fc_layer)

#%% # Get the number of input features to the original fully connected layer
# This is stored in the 'in_features' attribute of the linear layer
num_features = original_fc_layer.in_features

# Define the number of output classes for the new classification task
num_classes = 5

# Create a new fully connected layer (Linear layer)
new_fc_layer = nn.Linear(in_features=num_features, out_features=num_classes)

# Replace the original fully connected layer of resnet18_model with the new_fc_layer
resnet18_model.fc = new_fc_layer

print("Model's New Fully Connected Layer:")
print(resnet18_model.fc)

#%% Modular Block
# Instantiate the MobileNetV3 Small architecture and load the selected weights
mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1')

print(mobilenet_model)

#%% # Iterate through each parameter in model.features.parameters()
for feature_parameter in mobilenet_model.features.parameters():
    # Set the requires_grad attribute of each feature_parameter to False
    feature_parameter.requires_grad = False

#%% # Access the final classification layer of the model
last_classifier_layer = mobilenet_model.classifier[-1]

print("Model's Original Output Layer:")
print(last_classifier_layer)

#%% # Access the in_features attribute of last_classifier_layer
num_features = last_classifier_layer.in_features

# Define the number of output classes for the new classification task
num_classes = 10

# Create a new Linear layer for classification
new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)

# Replace the original last classification layer with the newly created layer
mobilenet_model.classifier[-1] = new_classifier

print("Model's New Output Layer:")
print(mobilenet_model.classifier[-1])

#%% # Define loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
# Only optimize the parameters that require gradients for mobilenet_model
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
                                   mobilenet_model.parameters()),
                            lr=0.001)

#%% Train the model!
# Number of epochs for new classifier head training
num_epochs = 5

# Start the training.
trained_model = helper_utils.training_loop(
    model=mobilenet_model, 
    trainloader=train_loader, 
    valloader=val_loader, 
    loss_function=loss_function, 
    optimizer=optimizer, 
    num_epochs=num_epochs, 
    device=device
)

#%% # Define a list of class names for the EMNIST digits (0-9).
emnist_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Visualize the model's predictions on the validation images
helper_utils.show_predictions(trained_model, val_loader, device, emnist_class_names)

#%% Strategy 2: Fine-Tuning (Train the Head and Top Layers)
print(trained_model)

#%% # The model from the previous training stage
fine_tune_model = trained_model

# Unfreeze the parameters of the last block in the 'features' section
for param in fine_tune_model.features[12].parameters():
    param.requires_grad = True
    
#%% # Verify that the parameters of an early block (e.g., features[0]) are frozen
print(f"Parameters in features[0] are frozen:       {not fine_tune_model.features[0][0].weight.requires_grad}")

# Verify that the parameters of a late block (e.g., features[12]) are now unfrozen
print(f"Parameters in features[12] are unfrozen:    {fine_tune_model.features[12][0].weight.requires_grad}")

# Verify that the classifier head remains unfrozen and trainable
print(f"Parameters in the classifier are unfrozen:  {fine_tune_model.classifier[-1].weight.requires_grad}")    

#%% # Create a new optimizer that targets all trainable parameters
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, fine_tune_model.parameters()),
    lr=1e-5  # A new, lower learning rate for fine-tuning
)

#%% 
# Number of epochs for the fine-tuning stage
num_epochs_fine_tune = 3

# Continue training the model
fine_tune_trained_model = helper_utils.training_loop(
    model=fine_tune_model,
    trainloader=train_loader,
    valloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs_fine_tune,
    device=device
)

#%%# ### Uncomment and execute the line below if you wish to visualize predictions

# Visualize the model's predictions on the validation images
helper_utils.show_predictions(fine_tune_trained_model, val_loader, device, emnist_class_names)

#%% Strategy 3: Full Retraining (Train the Entire Model)
# The model from the previous fine-tuning stage
full_retrain_model = fine_tune_trained_model

# Unfreeze all parameters in the model
for param in full_retrain_model.parameters():
    param.requires_grad = True

#%% 
# Verify that an early layer is now unfrozen and trainable
print(f"Parameters in features[0] are unfrozen: {full_retrain_model.features[0][0].weight.requires_grad}")

#%% # The optimizer now targets all parameters in the model
optimizer = torch.optim.SGD(
    full_retrain_model.parameters(),
    lr=1e-4
)

#%% 
# Number of epochs for the full retraining stage
num_epochs_full_retrain = 3

# Continue training the entire model
final_model = helper_utils.training_loop(
    model=full_retrain_model,
    trainloader=train_loader,
    valloader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs_full_retrain,
    device=device
)

#%% # ### Uncomment and execute the line below if you wish to visualize predictions

# # Visualize the model's predictions on the validation images
helper_utils.show_predictions(final_model, val_loader, device, emnist_class_names)