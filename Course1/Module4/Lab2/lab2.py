#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:32:19 2025

@author: ek
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet

import helper_utils
#%% Load dataset
dataset = helper_utils.get_dataset()

transform = transforms.ToTensor()
dataset.transform = transform

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

img_batch, label_batch = next(iter(dataloader))
print("Batch shape:", img_batch.shape)  # Should be [batch_size, 1, 28, 28]

#%% Define a simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Block
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Block
        # For Fashion MNIST: input images are 28x28,
        # after conv+pool: 32x14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 classes for Fashion MNIST

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x
    
#%% Instantiate the NN
simple_cnn = SimpleCNN()

try:
    output = simple_cnn(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass: {e}\033[0m")
    
#%% Add debug info to the model definition class
class SimpleCNNDebug(SimpleCNN):
    def __init__(self):
        super().__init__()
        # The super().__init__() call above properly initializes all layers from SimpleCNN
        # No need to redefine the layers here

    def forward(self, x):
        print("Input shape:", x.shape)
        print(
            " (Layer components) Conv layer parameters (weights, biases):",
            self.conv.weight.shape,
            self.conv.bias.shape,
        )
        x_conv = self.relu(self.conv(x))

        print("===")

        print("(Activation) After convolution and ReLU:", x_conv.shape)
        x_pool = self.pool(x_conv)
        print("(Activation) After pooling:", x_pool.shape)

        print(
            "(Layer components) Linear layer fc1 parameters (weights, biases):",
            self.fc1.weight.shape,
            self.fc1.bias.shape,
        )

        x_fc1 = self.relu_fc(self.fc1(x_pool))

        print("===")

        print("(Activation) After fc1 and ReLU:", x_fc1.shape)

        print(
            "(Layer components) Linear layer fc2 parameters (weights, biases):",
            self.fc2.weight.shape,
            self.fc2.bias.shape,
        )
        x = self.fc2(x_fc1)

        print("===")

        print("(Activation) After fc2 (output):", x.shape)
        return x
    
#%% Try again
simple_cnn_debug = SimpleCNNDebug()

try:
    output_debug = simple_cnn_debug(img_batch)  
except Exception as e:
    print(f"\033[91mError during forward pass in debug model: {e}\033[0m")
    
#%% Fix the model: Flattening operation was forgotten
class SimpleCNNFixed(SimpleCNN):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("Input shape:", x.shape)
        print(
            " (Neuron components) Conv layer parameters (weights, biases):",
            self.conv.weight.shape,
            self.conv.bias.shape,
        )
        x_conv = self.relu(self.conv(x))

        print("===")

        print("(Activation) After convolution and ReLU:", x_conv.shape)
        x_pool = self.pool(x_conv)
        print("(Activation) After pooling:", x_pool.shape)

        x_flattened = torch.flatten(
            x_pool, start_dim=1
        )  # Flatten all dimensions except batch
        print("(Activation) After flattening:", x_flattened.shape)

        print(
            "(Neuron components) Linear layer fc1 parameters (weights, biases):",
            self.fc1.weight.shape,
            self.fc1.bias.shape,
        )

        x_fc1 = self.relu_fc(self.fc1(x_flattened))

        print("===")

        print("(Activation) After fc1 and ReLU:", x_fc1.shape)

        print(
            "(Neuron components) Linear layer fc2 parameters (weights, biases):",
            self.fc2.weight.shape,
            self.fc2.bias.shape,
        )
        x = self.fc2(x_fc1)

        print("===")

        print("(Activation) After fc2 (output):", x.shape)
        return x
    
#%%# Fixed version
simple_cnn_fixed = SimpleCNNFixed()

output = simple_cnn_fixed(img_batch)

#%% The model is now working correctly, but the forward method is quite verbose and repetitive. To make the code cleaner and more modular, you can use nn.Sequential to define the convolutional and fully connected blocks.

# In this way you gain several advantages:

# Modularity: Each block is defined as a separate module, making it easier to understand and modify.
# Reusability: You can easily reuse the blocks in other models or experiments.
# Cleaner Code: The forward method becomes much simpler, as it only needs to call the blocks sequentially.
# Less Error-Prone: By defining the blocks in one place, you reduce the chances of making mistakes when implementing the forward method.

class SimpleCNN2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Block
        # For Fashion MNIST: input images are 28x28,
        # after conv+pool: 32x14x14
        flattened_size = 32 * 14 * 14
        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 classes for Fashion MNIST
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        x = self.fc_block(x)
        return x

#%% 
simple_cnn_seq = SimpleCNN2Seq()
output = simple_cnn_seq(img_batch)

print("Output shape from sequential model:", output.shape)

#%% Statistical Inspection of the Initialization
# A common check when inspecting a model is to look at the statistics of some activations to ensure that they are within a reasonable range.
class SimpleCNN2SeqDebug(SimpleCNN2Seq):
    def __init__(self):
        super().__init__()
        # The super().__init__() call above properly initializes all layers from SimpleCNN2Seq
        # No need to redefine the layers here

    def get_statistics(self, activation):
        mean = activation.mean().item()
        std = activation.std().item()
        min_val = activation.min().item()
        max_val = activation.max().item()

        print(f" Mean: {mean}")
        print(f" Std: {std}")
        print(f" Min: {min_val}")
        print(f" Max: {max_val}")
        return mean, std, min_val, max_val

    def forward(self, x):
        features = self.conv_block(x)
        x = torch.flatten(features, start_dim=1)  # Flatten all dimensions except batch

        print("After conv_block, the activation statistics are:")
        self.get_statistics(features)

        x = self.fc_block(x)
        print("After fc_block, the activation statistics are:")
        self.get_statistics(x)
        return x
    
#%% 
simple_cnn_seq_debug = SimpleCNN2SeqDebug()

for idx, (img_batch, _) in enumerate(dataloader):
    if idx < 5:
        print(f"=== Batch {idx} ===")
        output_debug = simple_cnn_seq_debug(img_batch)
        
#%% Model Inspection
# With the previous model working correctly, you will now inspect a pre-existing complex model from torchvision.models, such as SqueezeNet.

# In this section you will make use of the inspection utilities provided by PyTorch to explore the model's architecture, layers, and parameters. These inspection techniques are foundational for effective debugging and for making informed modifications to your neural network designs. 
# Load SqueezeNet model
complex_model = SqueezeNet()

print(complex_model)

#%% For complex models, printing the entire model architecture can be overwhelming. Instead, you can make use of named_children() and children() to iterate through the top-level blocks of the model.
# Iterate through the main blocks
for name, block in complex_model.named_children():
    print(f"Block {name} has a total of {len(list(block.children()))} layers:")
    
    # List all children layers in the block
    for idx, layer in enumerate(block.children()):
        # Check if the layer is terminal (no children) or not
        if len(list(layer.children())) == 0:
            print(f"\t {idx} - Layer {layer}")
        # If the layer has children, it's a sub-block, then print only the number of children and its name
        else:
            layer_name = layer._get_name()  # More user-friendly name
            print(f"\t {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")  

#%% You will now zoom into one of the Fire modules to see its internal structure.

#For that you can use modules() to iterate through all the layers and sub-modules of the model.
first_fire_module = complex_model.features[3]

for idx, module in enumerate(first_fire_module.modules()):
    # Avoid printing the top-level module itself
    if idx > 0 :
        print(module)
        
#%% Detail Inspection
# You will now count how many Conv2d layers are in the model.        
type_layer = nn.Conv2d

selected_layers = [layer for layer in complex_model.modules() if isinstance(layer, type_layer)]

print(f"Number of {type_layer.__name__} layers: {len(selected_layers)}")

#%% You will now count the total number of parameters in the model. This gives you an idea of the model's complexity and capacity.
# total number of parameters in the model
total_params = sum(p.numel() for p in complex_model.parameters())
print(f"Total number of parameters in the model: {total_params}")

#%% Now, you can take this further by inspecting the parameters of each terminal layer (layers without children) in the model. For each terminal layer, you will print its name and the total number of parameters it contains. This helps to identify which layers contribute most to the model's parameter count and can be useful for model optimization, pruning, or understanding where the model's capacity lies.
counting_params = {}

# For each terminal layer print its number of parameters
for layer in complex_model.named_modules():
    n_children = len(list(layer[1].children()))
    if n_children == 0:  # Terminal layer
        layer_name = layer[0]
        n_parameters = sum(p.numel() for p in layer[1].parameters())
        counting_params[layer_name] = n_parameters
        print(f"Layer {layer_name} has {n_parameters} parameters")

# Plotting the distribution of parameters per layer
helper_utils.plot_counting(counting_params)

#%% Conclusion¶
# You have now successfully debugged, refactored, and inspected PyTorch models. In this lab, you saw firsthand that a model's forward pass is not a black box and that by strategically adding print statements, you can diagnose and solve common but frustrating errors like shape mismatches.

# You have moved beyond simply writing model code and can now make it more robust and readable by grouping layers into logical blocks with nn.Sequential. This practice of modularization makes your architectures easier to understand, reuse, and adapt. You also learned how to perform essential sanity checks by inspecting activation statistics and how to systematically explore any PyTorch model, no matter how complex, using inspection utilities like .modules() and .named_children().

# With these fundamental skills of debugging and inspection, you are well-prepared for more advanced challenges.