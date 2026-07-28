# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:18:07 2026

C3 M4 L4. A Practical Guide to Model Quantization in PyTorch.

Deploying deep learning models often requires optimizing them for efficiency, especially in environments with limited resources. While models are typically trained using high-precision 32-bit floating-point numbers to capture the small, exact adjustments needed during learning, this level of precision is not always necessary for inference once the model is trained.

Model quantization addresses this by converting a model's weights and activations to a lower-precision format, like 8-bit integers, leading to smaller model sizes and faster performance.

In this notebook, you will embark on a hands-on journey through the landscape of quantization with PyTorch. You will begin by establishing a performance benchmark with a standard floating-point model. Then, you will explore and apply three distinct strategies, each building on the last, to see their effects in practice.

    Dynamic Quantization: You'll start with a straightforward, post-training technique that provides an immediate size reduction with minimal code by converting weights to integers and quantizing activations on-the-fly during inference.
    Static Quantization: Next, you'll implement a more involved post-training method that can yield better performance. This technique involves a calibration step, where you run sample data through the model to determine the best way to quantize the activations ahead of time.
    Quantization-Aware Training (QAT): Finally, you'll use the most advanced technique, which simulates quantization effects during a fine-tuning phase. This allows the model to adapt its weights to the rounding noise, helping to achieve the highest possible accuracy in the final quantized model.

To see these concepts applied in a practical scenario, an optional section will guide you through quantizing a large, pre-trained Visual Question Answering (VQA) model. Through this process, you will gain the practical knowledge to evaluate trade-offs and choose the right optimization strategy for your own projects.

@author: ekirshin
"""
from IPython.display import Image as DisplayImage
import torch
import torch.nn as nn
import torch.quantization
from tqdm.auto import tqdm

import helper_utils

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")

# %% Baseline Model
# =============================================================================
# Before applying advanced optimization techniques like quantization, it is crucial to first establish a benchmark. This benchmark is your baseline model — the original, fully-trained, and un-optimized version of the network.
# 
# By measuring its key performance characteristics, such as model size and inference speed, you create a standard reference point. This baseline is essential because it allows you to concretely measure the effectiveness of your optimizations. Later, as you apply different quantization techniques, you will compare the results of each new model directly against this baseline to clearly see the improvements and any potential trade-offs.
# CNN Model Architecture
# 
#     Define the architecture for the Convolutional Neural Network (CNN) that will be used as a baseline throughout this lab.
#     The architecture includes:
#         A sequence of torch.nn.Conv2d layers for feature extraction, each followed by a torch.nn.BatchNorm2d layer to stabilize learning.
#         torch.nn.MaxPool2d layers to downsample the feature maps after each convolutional block.
#         torch.nn.Dropout applied between the fully connected layers to reduce overfitting.
#         A final classification head made of three torch.nn.Linear layers.
#         The forward method orchestrates the flow of data through these layers.
# =============================================================================
class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) implementation for image classification.
    """
    def __init__(self):
        """
        Initializes the CNN layers including convolution, batch normalization, 
        pooling, dropout, and fully connected layers.
        """
        super(CNN, self).__init__()
        # Define the convolutional and batch normalization layers 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Define a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Define a dropout layer
        self.dropout = nn.Dropout(0.2)
        # Define the fully connected layers for the classification head 
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # Define the ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass logic of the model.

        Args:
            x (torch.Tensor): The input image batch to be processed.

        Returns:
            x (torch.Tensor): The output logits representing class scores.
        """
        # Pass input through the sequence of convolutional blocks 
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Flatten the output from the convolutional layers for the fully connected layers 
        x = x.view(-1, 512 * 2 * 2) 
        # Pass data through the fully connected layers 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        # Map features to the final class outputs
        x = self.fc3(x) 
        
        return x

# %% Loading the Dataset
# Call the helper function, load_cifar10, to download and prepare the CIFAR10 dataset.
# It creates and returns trainloader and testloader.
trainloader, testloader = helper_utils.load_cifar10()

# %% Model Training (Optional)
# =============================================================================
# The next logical step is to train the CNN model on the CIFAR10 dataset. However, training can be time-consuming, taking approximately 10 minutes for 30 epochs in this GPU environment.
# To save time, you are provided a pre-trained model that has already been trained. Below are the logs from its training session. 
# =============================================================================
# =============================================================================
# The best results achieved were: a validation loss of 0.4086 and an accuracy of 88.20%.
# 
#     Validation Loss: 0.4086
#     Validation Accuracy: 88.20%
# =============================================================================
# Initialize the model
model = CNN()

# Set number of training epochs
num_epochs = 30

# Run the training loop
helper_utils.training_loop(model, trainloader, testloader, num_epochs, DEVICE)

# %% Load the Pre-trained Model
# =============================================================================
# Define the path to the provided pretrained model, cifar10_cnn_30_epochs_best.pt.
# 
#     NOTE: If you trained your own model in the optional cell above, please make sure to set the path to baseline_model_path = 'cifar10_cnn_best.pt', which is the name used by the training script for the saved model.
# 
# =============================================================================
#baseline_model_path = './baseline_pretrained_model/cifar10_cnn_30_epochs_best.pt'
baseline_model_path = 'cifar10_cnn_best.pt'

# %% Create an instance of the CNN model.
# Load the pre-trained weights from the baseline_model_path into the model structure.
# Switch the model to evaluation mode, which is a necessary step for the quantization APIs to work correctly.
# Create an instance of the CNN model architecture
baseline_model = CNN()
# Load the pre-trained weights from the file specified in baseline_model_path
baseline_model.load_state_dict(torch.load(baseline_model_path))
# Set the model to evaluation mode
baseline_model.eval()

# %% Baseline Performance Metrics
# Calculate the model's size and inference time to establish the baseline performance metrics for the quantization comparisons.
# Calculate the model's size in megabytes (MB)
baseline_model_size = helper_utils.get_model_size(baseline_model)
# Measure the average inference time in milliseconds (ms)
baseline_model_inf_time = helper_utils.measure_average_inference_time_ms(baseline_model)

# Print the metrics
print(f"Baseline model size: {baseline_model_size:.2f} MB")
print(f"Baseline model inference time: {baseline_model_inf_time:.2f} ms")

# %% Dynamic Quantization
# =============================================================================
# With the baseline established, you can now begin optimizing the model. The first and simplest approach you will explore is Dynamic Quantization.
# 
# This strategy is straightforward: it involves converting only the weights in the model from the standard 32-bit floating-point format to a more efficient, lower-precision format like 8-bit integers (INT8). The activations, on the other hand, are calculated in floating-point during inference and are quantized on-the-fly as they are passed to the quantized weight operations. This method is easy to apply and can yield significant benefits, such as reducing the model's size by half and speeding up inference time.
# 
# Before you apply this technique, it's useful to inspect the current data types of the model's weights. The following code will print the dtype for each Conv2d and Linear layer, giving you a clear "before" picture that confirms all weights are in the standard torch.float32 format.
# 
#     Print the dtype for each Conv2d and Linear layer, giving you a clear "before" picture that confirms all weights are in the standard torch.float32 format.
# =============================================================================
print("--- Weight dtypes before quantization ---")
# Iterate through the model's layers
for name, module in baseline_model.named_modules():
    # Check if the layer is a Conv2d or Linear layer
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight.dtype}")

# %% Applying Dynamic Quantization
# =============================================================================
#     Use the torch.quantization.quantize_dynamic function to perform this operation.
#         baseline_model: The first argument is the floating-point model you want to quantize.
#         {nn.Linear, nn.Conv2d}: This specifies the set of layer types you want to dynamically quantize.
#         dtype=torch.qint8: This tells the function the target data type for the quantized weights.
#     Save the state dictionary of the newly created quantized model.
# =============================================================================
# Apply dynamic quantization
quantized_dynamic_model = torch.quantization.quantize_dynamic(
    # Model to be quantized. 
    baseline_model,
    # The layers to quantize
    {nn.Linear, nn.Conv2d},
    # The target data type for the quantized weights
    dtype=torch.qint8
)

# Save the state dictionary of the newly quantized model. 
torch.save(quantized_dynamic_model.state_dict(), 'cifar10_cnn_quantized_dynamic.pth')

# %% 
# =============================================================================
# After quantization, the specified layers are swapped out for new, dynamically quantized versions. This means:
# 
#     Your previous torch.nn.Linear layers are now instances of torch.nn.quantized.dynamic.Linear.
#     Similarly, any torch.nn.Conv2d layers would become torch.nn.quantized.dynamic.Conv2d.
# 
# You can now inspect the data types of these new layers to confirm the change.
# =============================================================================
print("--- Weight dtypes after dynamic quantization ---")
# Iterate through the quantized model's layers
for name, module in quantized_dynamic_model.named_modules():
    # Check if the layer is a dynamically quantized Conv2d or Linear layer
    if isinstance(module, (torch.nn.quantized.dynamic.Conv2d, torch.nn.quantized.dynamic.Linear)):
        # For quantized layers, the weight is packed and must be accessed as a method
        print(f"Layer: {name:<10} | Weight dtype: {module.weight().dtype}")

# =============================================================================
# Notice something interesting? Even though you passed nn.Conv2d layers to the function, only the Linear layers were actually converted to the torch.qint8 data type. This is because PyTorch's dynamic quantization implementation is highly optimized for operations where the weights are the primary bottleneck, which is most often the case with Linear layers.
# 
# This isn't a limitation of quantization overall. The more advanced techniques you'll explore next, Static Quantization and Quantization-Aware Training (QAT), are designed to handle and quantize Conv2d layers effectively, which is essential for getting the best performance out of CNN architectures.
# =============================================================================

# %% Compare Performance
# Now that you've applied dynamic quantization, measure the size and inference speed of the new model compared with the baseline model.
# Calculate the model's size in megabytes (MB)
quantized_dynamic_model_size = helper_utils.get_model_size(quantized_dynamic_model)
# Measure the average inference time in milliseconds (ms)
quantized_dynamic_model_inf_time = helper_utils.measure_average_inference_time_ms(quantized_dynamic_model)

# Generate the Markdown comparison table
helper_utils.comparison_table(
    baseline_model_size=baseline_model_size,
    baseline_model_time=baseline_model_inf_time,
    quantized_model_size=quantized_dynamic_model_size,
    quantized_model_time=quantized_dynamic_model_inf_time,
    quantization_type="Dynamic"
)

# %% Static Quantization
# =============================================================================
# Next, you will explore Static Quantization, a more powerful but also more involved optimization technique compared to its dynamic counterpart.
# 
# The key difference is that static quantization converts both the model's weights and its activations to a lower-precision integer format, such as INT8. Because activations are also being quantized, the process requires an extra, critical step: calibration.
# 
# The following cells will guide you through the complete workflow:
# Statically Quantized CNN Architecture
# 
#     Define a new QuantizedCNN class with two key additions:
#         self.quant = torch.quantization.QuantStub(): This is a "quantization stub" module that you will place at the very beginning of your forward pass. Its job is to convert the incoming floating-point tensors into quantized tensors.
#         self.dequant = torch.quantization.DeQuantStub(): This is a "dequantization stub" that you will place at the very end. It converts the quantized output tensors back into floating-point format.
#     The rest of the architecture remains identical to the original CNN.
# =============================================================================
class QuantizedCNN(nn.Module):
    """
    A Convolutional Neural Network designed for quantization-aware training or static quantization.

    Attributes:
        quant (QuantStub): Module to convert floating-point tensors to quantized tensors.
        conv1 (Conv2d): First convolutional layer.
        bn1 (BatchNorm2d): First batch normalization layer.
        conv2 (Conv2d): Second convolutional layer.
        bn2 (BatchNorm2d): Second batch normalization layer.
        conv3 (Conv2d): Third convolutional layer.
        bn3 (BatchNorm2d): Third batch normalization layer.
        conv4 (Conv2d): Fourth convolutional layer.
        bn4 (BatchNorm2d): Fourth batch normalization layer.
        pool (MaxPool2d): Max pooling layer for spatial downsampling.
        relu (ReLU): Rectified Linear Unit activation function.
        fc1 (Linear): First fully connected layer.
        fc2 (Linear): Second fully connected layer.
        fc3 (Linear): Final fully connected layer for classification.
        dequant (DeQuantStub): Module to convert quantized tensors back to floating-point.
    """
    def __init__(self):
        """
        Initializes the model architecture with quantization stubs and standard CNN layers.
        """
        super(QuantizedCNN, self).__init__()
        # Module to transform floating-point input into a quantized format
        self.quant = torch.quantization.QuantStub()

        # Primary feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Spatial reduction and activation layers
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Classification head layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        # Module to transform quantized output back into floating-point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        """
        Defines the computation performed at every call, including quantization and dequantization steps.

        Args:
            x (torch.Tensor): The input floating-point tensor containing a batch of images.

        Returns:
            x (torch.Tensor): The output floating-point tensor representing classification logits.
        """
        # Convert the input tensor from floating-point to quantized
        x = self.quant(x)

        # Execute the convolutional feature extraction sequence
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Reshape the spatial feature maps into a vector for linear layers
        x = x.reshape(-1, 512 * 2 * 2)
        # Pass the features through the fully connected classification head
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Convert the resulting tensor from quantized back to floating-point
        x = self.dequant(x)
        
        return x

# %% Prepare Model for Static Quantization
# =============================================================================
# Perform the initial setup required before you can calibrate and convert the model.
# 
#     First, create an instance of your new QuantizedCNN class.
#     Next, copy the weights from the original trained baseline_model into this new quantization-ready instance.
#     Set the model to evaluation mode.
#     Finally, you attach a quantization configuration (qconfig) to the model. Here, you use torch.quantization.get_default_qconfig('x86'), which is the recommended default configuration for server-side inference on x86 CPUs. This step also implicitly prepares the model by inserting "observer" modules that will be used to analyze data flow in the next step.
#         NOTE: While the older 'fbgemm' backend is still available, 'x86' is the new recommended default.
# 
# =============================================================================
# Create an instance of the new QuantizedCNN class
quantized_static_model = QuantizedCNN()

# Copy the learned weights from the pre-trained baseline_model into the new quantized_static_model
quantized_static_model.load_state_dict(baseline_model.state_dict())
# Set the model to evaluation mode
quantized_static_model.eval()

# Set the quantization configuration for the model. 
# 'fbgemm' is a configuration optimized for server-side inference on x86 CPUs. 
# This also attaches observer modules that will be used during calibration. 
quantized_static_model.qconfig = torch.quantization.get_default_qconfig('x86')

# %% Prepare and Calibrate
# Use torch.quantization.prepare.
# This function takes the model with the attached qconfig and formally prepares it for calibration by activating the observer modules that were inserted in the previous step. These observers are now ready to watch the data that flows through the model.
# Prepare model for quantization
torch.quantization.prepare(quantized_static_model, inplace=True)

# %%


