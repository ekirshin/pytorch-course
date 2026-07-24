# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:32:12 2026

C3 M4 L3. Introduction to Pruning with PyTorch.

Creating a highly accurate model is a significant achievement, but for many real-world applications, accuracy alone is not enough. A model must also be fast, lightweight, and efficient to run on devices with limited memory or processing power, such as mobile phones. This is where model compression techniques become essential, allowing you to trim down your model without a major loss in performance.

Pruning is a powerful compression method that operates on a simple principle: not all parts of a trained network are equally important. Many of the millions of parameters in a large model contribute very little to the final predictions. By systematically removing these less useful connections, you can create a more streamlined and efficient network.

This lab provides a hands-on guide to implementing various pruning strategies using PyTorch's torch.nn.utils.prune module. Through practical examples, you will learn how to apply these techniques, verify their effects, and make the resulting sparsity permanent in your models. You will explore three fundamental approaches:

    Unstructured Pruning: Removing individual weights based on their magnitude, creating a sparse model without altering its architecture.

    Structured Pruning: Removing entire structural units, such as channels in a convolutional layer or neurons in a linear layer, which can lead to direct speedups.

    Global Pruning: Applying a single pruning criterion across multiple layers at once to achieve a target sparsity for the entire model.

Additionally, an optional section will guide you through an end-to-end pipeline of applying global pruning to a pre-trained ResNet18 model. By comparing its performance against an unpruned baseline, you will see firsthand how a significantly sparser network can achieve comparable results.

@author: ekirshin
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import helper_utils

# %% Model Definition
# =============================================================================
# To demonstrate various pruning techniques, you'll define a SimpleModel for this purpose.
# 
#     Demonstration Only: This model will not be used for training on real data. It is a dummy architecture designed solely to illustrate pruning mechanics on a small, manageable scale.
#     Key Layers: It contains the two layers you will target for pruning: a torch.nn.Conv2d layer (conv1) and a torch.nn.Linear layer (fc1).
#     Strict Input Requirement: The model is hard-coded to accept an input size of pixels.
#     Dimension Logic: The fully connected layer is initialized as nn.Linear(16 * 4 * 4, 10) based on standard convolution arithmetic:
#         A 6x6 input processed by a 3x3 kernel (with no padding) results in a 4x4 spatial output.
#         Flattening this output (16 channels x 4 x 4) requires exactly 256 inputs for the linear layer.
#     Output Assumption: The final layer produces 10 output features, assuming a classification task with 10 distinct classes.
# 
# =============================================================================
# Create a simple model for demonstration
class SimpleModel(nn.Module):
    """
    A simple convolutional neural network defined for demonstration purposes.
    """
    def __init__(self):
        """
        Initializes the SimpleModel with a convolutional layer and a fully connected layer.
        """
        # Call the parent class initializer
        super(SimpleModel, self).__init__()
        # Define the first convolutional layer with 3 input channels and 16 output channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # Define the ReLU activation function
        self.relu1 = nn.ReLU()
        # Define the fully connected layer
        # The input size is calculated based on a 6x6 input image: (6-3+1) = 4x4 spatial output
        self.fc1 = nn.Linear(16 * 4 * 4, 10)  # Assuming input is 6x6

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor containing the image data.

        Returns:
            The output tensor after passing through the network.
        """
        # Apply the convolutional layer followed by the ReLU activation
        x = self.relu1(self.conv1(x))
        # Flatten the tensor dimensions for the fully connected layer
        x = x.view(x.size(0), -1)
        # Pass the flattened tensor through the fully connected layer
        x = self.fc1(x)
        return x

# %% Unstructured Pruning
# Unstructured pruning is a technique used to reduce a model's size by removing individual weights that have a minimal impact on performance, regardless of their position in the model's architecture. The core idea is that many weights in a trained network are often very close to zero and don't contribute significantly to the output. By removing them, you can create a smaller, more efficient model.

# %% Applying Unstructured Pruning
# Initialize the model from SimpleModel class to perform Unstructured Pruning.
model = SimpleModel()

# %%
# =============================================================================
# Before applying pruning, you'll first inspect the initial state of your model instance. This baseline will help you see exactly what changes after you apply a pruning technique.
# 
#     Model Parameters: Lists of all the learnable parameters in the model. You will see the initial weight and bias for both the conv1 and fc1 layers.
#     Model Buffers: List of the model's buffers. Buffers are stateful tensors within the model that are not considered parameters to be trained by the optimizer (for example, the running mean and variance in a batch normalization layer).
#         You are checking the buffers because, as you'll see later, the pruning process adds a weight_mask as a buffer to track which weights have been zeroed out.
#         Initially, this list of buffers will be empty.
# =============================================================================
# Check state before pruning
print("BEFORE UNSTRUCTURED PRUNING:\n")

# Model Parameters
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
# Model Buffers
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

# %% Print out the raw numerical values of a kernel from the conv1 layer.
# This provides a direct, unambiguous baseline that you can compare against after the pruning is applied.
print("--- conv1 Weights Before Pruning ---\n")
helper_utils.show_weights(model, ['conv1'])

# %% Now, this is where you'll perform the actual pruning operation on your model.
# =============================================================================
#  Use the prune.l1_unstructured function to apply pruning to the conv1 layer.
#         The function identifies and removes individual weights based on their L1 norm (their absolute value), targeting the least important ones.
#         model.conv1: This specifies the target module — the first convolutional layer in your model.
#         name="weight": This tells the function to prune the weight tensor within that layer.
#         amount=0.3: This sets the fraction of weights to remove, meaning 30% of the connections in the conv1 layer will be pruned.
# =============================================================================
# Apply unstructured pruning to conv1
prune.l1_unstructured(model.conv1, name="weight", amount=0.3)

# %% Verifying Unstructured Pruning
# =============================================================================
# To verify the results of the pruning, you'll inspect both the model's internal structure and its numerical weights.
# 
#     Check the Model State:
#         Expected Buffers: While this list was empty before pruning, you'll now see conv1.weight_mask appear in the model's buffers list. This new buffer is the tensor of 0s and 1s that PyTorch uses to track which weights have been removed.
#         Expected Parameters: You'll notice the original conv1.weight is gone from the parameters list and has been replaced by conv1.weight_orig. This new parameter holds the underlying, un-masked weight values.
#     Check the Numerical Weights:
#         Expected Weights: In the numerical output, you will now clearly see several 0. values scattered throughout the matrix. These are the individual weights that were zeroed out by the pruning function because they had the lowest absolute values.
# =============================================================================
# Check state after pruning
print("AFTER UNSTRUCTURED PRUNING:\n")
print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

# --- Show numerical weights after pruning ---
print("\n--- conv1 Weights After Pruning ---\n")
helper_utils.show_weights(model, ['conv1'])

# %% Check Pruning Statistics and Types: This final check dives deeper by looking at the exact sparsity achieved and the change in the weight attribute's type. Sparsity is simply the percentage of weights that are zero; a higher sparsity means more of the layer has been pruned.
# =============================================================================
#   Pruning Statistics: By counting the zeros in the weight_mask, the code confirms the Sparsity is ~30%, matching the amount you requested.
#   Weight Types: This inspection reveals that weight_orig is now the true, learnable Parameter, while .weight has become a non-learnable Tensor that is dynamically computed from weight_orig and weight_mask.
# =============================================================================
# Check if the pruning mask exists.
if hasattr(model.conv1, 'weight_mask'):
    
    # Count weights where the mask is 0.
    pruned_count = torch.sum(model.conv1.weight_mask == 0).item()
    
    # Get the total number of weights.
    total_count = model.conv1.weight_mask.numel()
    
    # Print statistics.
    print(f"\nPruning statistics:")
    print(f"Total weights in conv1: {total_count}")
    print(f"Pruned weights (set to zero) in conv1: {pruned_count}")
    print(f"Sparsity: {pruned_count/total_count:.2%}")

    # Verify the change in weight attribute types.
    print(f"\nWeight type: {type(model.conv1.weight)}")
    print(f"Weight_orig type: {type(model.conv1.weight_orig)}")
else:
    # Handle pruning failure.
    print("Pruning did not work as expected! No weight_mask found.")

# %% Making Unstructured Pruning Permanent
# =============================================================================
# You've successfully applied pruning to the conv1 layer and verified that roughly 30% of its weights are now zero. However, the process isn't finished yet. It's important to make this pruning permanent because, right now, your model is in a temporary "pruned state" and is carrying extra baggage.
# 
# As you saw when you inspected the model's parameters and buffers, it's holding onto both the original weights (in the conv1.weight_orig parameter) and the conv1.weight_mask buffer. This means that during every forward pass, it has to compute the sparse weights on-the-fly by multiplying these two tensors.
# 
# Calling prune.remove() cleans this up. It removes the mask and the original parameter, saving the final, sparse weight tensor back into the standard .weight attribute. This removes the computational overhead and simplifies the model back to a standard structure, making it truly efficient and ready for deployment.
# 
#     Use the prune.remove() utility to finalize the conv1 layer.
#         This function removes the weight_mask and weight_orig attributes from the module.
#         It replaces them with a single, final weight parameter that contains the sparse (zeroed-out) weights, making the model's structure clean and efficient again.
# =============================================================================
# Make pruning permanent for conv1
prune.remove(model.conv1, 'weight')

# %% Confirm that the prune.remove() function worked correctly and the model has been returned to a standard, clean state.
# =============================================================================
# Model Parameters: You'll see that the conv1.weight_orig parameter is gone, and the standard conv1.weight has returned to the list of parameters. This shows the model's structure is back to normal.
# Model Buffers: The conv1.weight_mask has been removed, and the buffers list is now empty again, just as it was before you started pruning. This confirms that all the pruning-related overhead is gone.
# prune.is_pruned: This confirms the pruning is permanent. The layer has returned to a standard module that contains sparse (zeroed-out) weights, and the temporary pruning machinery has been removed.
#     This would have returned True if you had called it before using prune.remove(), as the layer was still in its temporary "pruned state" at that point. 
# =============================================================================
# Check state to see if pruning is permanent
print("AFTER MAKING UNSTRUCTURED PRUNING PERMANENT:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")
print(f"\nIs conv1 still considered pruned? {prune.is_pruned(model.conv1)}\n")

helper_utils.show_weights(model, ['conv1'])

# %% Structured Pruning
# =============================================================================
# Now, you'll explore structured pruning. Unlike unstructured pruning which removes individual weights, this technique removes entire structural units — in this case, entire neurons from a linear layer. This approach can lead to more significant improvements in computational efficiency without requiring specialized hardware.
# 
# You'll apply structured pruning to the fc1 layer of the SimpleModel.
# Applying Structured Pruning
# 
#     First, you'll re-initialize the model to demonstrate structured pruning on a fresh, un-pruned instance.
# =============================================================================
model = SimpleModel()

# %% Next, you'll inspect the initial state of the fc1 layer's weights before applying any changes.
# Check state before pruning
print("BEFORE STRUCTURED PRUNING:\n")
# Model State
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

print("\n--- fc1 Weights Before Pruning ---\n")
helper_utils.show_weights(model, ['fc1'])

# %% Now you'll apply the structured pruning using the prune.ln_structured function.
# =============================================================================
#     This function removes entire rows or columns based on their Ln-norm.
#     You're targeting the fc1 layer and removing 50% (amount=0.5) of its neurons.
#     n=2: specifies using the L2 norm (the vector's magnitude or Euclidean distance) to determine which neurons are least important.
#     dim=0, is crucial: for a linear layer's weight matrix of shape [output_features, input_features], this prunes along dimension 0, removing entire output neurons.
# =============================================================================
# Apply structured pruning to fc1
prune.ln_structured(model.fc1, name="weight", amount=0.5, n=2, dim=0)

# %% Verifying Structured Pruning
# =============================================================================
# To verify the results, you'll inspect the model's state and weights again.
# 
#     As before, you'll see a weight_mask and weight_orig have been created, but this time for the fc1 layer.
#     The key difference to look for in the numerical output is that entire rows of the fc1 weight matrix are now zero, corresponding to the pruned neurons.
# =============================================================================
# Check state after pruning
print("AFTER STRUCTURED PRUNING:\n")
print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

# --- Show numerical weights after pruning ---
print("\n--- fc1 Weights After Pruning ---\n")
helper_utils.show_weights(model, ['fc1'])

# %% This check calculates the pruning statistics.
# Note that instead of counting individual zero-weights, this code now counts how many entire rows are zero to confirm that 50% of the output neurons were removed.
# Check if the pruning mask exists for fc1.
if hasattr(model.fc1, 'weight_mask'):
    # For structured pruning, you count entire rows of zeros.
    # Determine if each row in the weight tensor is all zeros.
    zero_rows = torch.sum(model.fc1.weight == 0, dim=1) == model.fc1.weight.shape[1]
    # Count the number of all-zero rows.
    pruned_count = torch.sum(zero_rows).item()
    # Get the total number of rows (output neurons).
    total_count = model.fc1.weight.shape[0]
    
    # Print the statistics.
    print(f"\nPruning statistics:\n")
    print(f"Total output neurons in fc1: {total_count}")
    print(f"Pruned output neurons (set to zero) in fc1: {pruned_count}")
    print(f"Sparsity: {pruned_count/total_count:.2%}")
    
    # Verify the change in attribute types.
    print(f"\nWeight type: {type(model.fc1.weight)}")
    print(f"Weight_orig type: {type(model.fc1.weight_orig)}")
else:
    # Handle pruning failure.
    print("\nStructured pruning did not work as expected!")

# %% Making Structured Pruning Permanent
# Finally, you'll make the structured pruning permanent. This process is the same as before: it removes the pruning mask and finalizes the model's weights.
# Make structured pruning permanent for fc1
prune.remove(model.fc1, 'weight')

# %% This final check confirms the model's parameters and buffers have returned to their normal state and that the fc1 layer is no longer considered "pruned" by PyTorch.
# Check if structured pruning is permanent
print("AFTER MAKING STRUCTURED PRUNING PERMANENT:\n")
print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")
print(f"\nIs fc1 still considered pruned? {prune.is_pruned(model.fc1)}\n")

helper_utils.show_weights(model, ['fc1'])

# %% Global Pruning
# =============================================================================
# The final technique you'll explore is global pruning. Unlike the previous methods where you set a pruning amount for each layer individually, global pruning considers all the specified layers together. It removes a percentage of the total number of weights from all layers combined, targeting the weakest connections across the entire model.
# 
# This often yields better performance because the model can choose to prune more heavily from layers that are less sensitive (like fc1) and less from layers that are more sensitive (conv1), rather than being forced to prune each by a fixed amount.
# Applying Global Pruning
# 
#     As before, you'll start by re-initializing the model to ensure a clean slate.
# =============================================================================
model = SimpleModel()

# %% Next, you'll inspect the initial state of both the conv1 and fc1 layers before applying any changes.
# Check the model's initial state before pruning.
print("BEFORE GLOBAL PRUNING:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

# Show the initial weights of both layers.
print("\n--- Initial Weights of conv1 and fc1 ---\n")
helper_utils.show_weights(model, ['conv1', 'fc1'])

# %%
# =============================================================================
# Now you'll apply global unstructured pruning across both the conv1 and fc1 layers.
# 
#     First, you define a tuple of (module, 'weight') pairs that specifies all the parameters to be considered for pruning.
# 
# The prune.global_unstructured function will then remove the weakest 20% (amount=0.2) of weights from this entire collection of parameters.
# 
#     This utilizes the prune.L1Unstructured method as its pruning criterion. By leveraging the L1 unstructured method in the global context, you apply this principle across all specified layers, thereby achieving a uniform level of sparsity throughout the network.
# =============================================================================
# Define the collection of parameters to be pruned globally.
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.fc1, 'weight'),
)

# Apply global unstructured pruning across the specified parameters.
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# %% Verifying Global Pruning
# =============================================================================
#     To verify the results, you'll inspect the model's state and weights again.
#         You should now see _orig and _mask attributes for both conv1 and fc1.
#     The numerical outputs will show scattered zeros in both layers.
# =============================================================================
# Verify the model's state after pruning.
print("AFTER GLOBAL PRUNING:\n")

print(f"Model parameters:", [name for name, _ in model.named_parameters()])
print(f"Model buffers:", [name for name, _ in model.named_buffers()])

print("\n--- Weights After Global Pruning ---\n")
helper_utils.show_weights(model, ['conv1', 'fc1'])

# %% This next check calculates the sparsity for each layer individually.
# You will likely see that the sparsity for conv1 and fc1 are not equal to each other or to 20%. This demonstrates the "global" nature of the pruning — it removed the weakest weights regardless of which layer they were in.
# Check global pruning results.
print("--- Sparsity per Layer After Global Pruning ---\n")

# Verify sparsity in each layer individually.
conv1_sparsity = 100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement())
fc1_sparsity = 100. * float(torch.sum(model.fc1.weight == 0)) / float(model.fc1.weight.nelement())

print(f"Sparsity in conv1: {conv1_sparsity:.2f}%")
print(f"Sparsity in fc1: {fc1_sparsity:.2f}%")

# %% Making Global Pruning Permanent
# The final step is to make the changes permanent for all layers that were part of the global pruning.
# Make the pruning permanent for all pruned layers by iterating through the list.
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)

# %% This final check confirms that all pruned layers have been restored to their standard structure.
print("\nAFTER MAKING GLOBAL PRUNING PERMANENT:\n")

print(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
print(f"Model buffers: {[name for name, _ in model.named_buffers()]}")

print(f"\nIs conv1 still considered pruned? {prune.is_pruned(model.conv1)}")
print(f"Is fc1 still considered pruned? {prune.is_pruned(model.fc1)}\n")

helper_utils.show_weights(model, ['conv1', 'fc1'])

# %%
# =============================================================================
# Now that you have explored these pruning techniques in detail, you might be wondering when and in what scenarios each method can be applied effectively in your work. Understanding the strengths and suitable contexts for each technique is crucial to maximizing their benefits in your neural network optimization efforts. Let's delve into some best practices and use cases for unstructured, structured, and global pruning, helping you decide which approach aligns best with your specific needs and goals.
# 
#     Unstructured Pruning:
#         Best Practices: Unstructured pruning is often used when you want to maximize sparsity while maintaining flexibility. It is best suited for models where the architecture should remain unchanged, and you only need to prune less important weights.
#         Use Cases: This technique is useful in scenarios where hardware support for sparsity is present, allowing the sparsely pruned model to be efficiently executed. It's suitable for environments like server-side deployments where computational resources are relatively abundant but memory savings are crucial.
#     Structured Pruning:
#         Best Practices: Use structured pruning when you need to improve inference speed and reduce the model's memory footprint effectively. This technique can significantly accelerate the model since entire structures like channels or neurons are removed.
#         Use Cases: It's ideal for edge devices or mobile applications where computational resources are limited, as structured pruning can help maintain a balance between model size and accuracy while enhancing computational efficiency.
#     Global Pruning:
#         Best Practices: Global pruning is advantageous when aiming for a consistent level of sparsity across multiple layers. It minimizes the risk of over-pruning critical parts of the model by evaluating the importance of weights globally rather than locally.
#         Use Cases: This approach is beneficial in situations where the overall model performance is more critical than the performance of individual layers. It's particularly effective for creating uniformly sparse models that can adapt to various processing environments without heavily compromising accuracy.
# 
# By understanding these techniques and their implications, you can tailor your pruning strategy to best fit your application's needs, whether it's speeding up inference, reducing memory usage, or balancing efficiency with accuracy.
# 
# In the optional section below, you'll get a hands-on opportunity to apply one of these techniques in a real-world scenario. Specifically, you will apply global pruning to a pre-trained model and observe the significant before-and-after results, providing a practical demonstration of its impact.
# =============================================================================

# %% (Optional) Practical Application: Global Pruning and Comparative Analysis of a Pre-trained Model
# =============================================================================
# This optional section transitions from theoretical understanding to a hands-on demonstration of model pruning in a more realistic setting. Here, you will observe a practical pipeline for applying global pruning to a pre-trained ResNet18 model. You'll see the impact of global pruning by comparing the performance results and model characteristics of a pruned ResNet18 against its unpruned counterpart, both trained on the same dataset for the same number of epochs. The goal is to provide a concrete example of how pruning can be integrated into a typical deep learning optimization workflow, demonstrating that comparable performance can be achieved on a pruned model, thus showcasing its benefits.
# 
# You might be wondering: among the various pruning techniques, why opt for global pruning, especially when working with a powerful pre-trained model like ResNet18?
# 
# Global pruning is chosen for this practical application with a pre-trained model like ResNet18 for several strategic reasons:
# 
#     Consistent Sparsity Across Layers: Global pruning is advantageous when aiming for a consistent level of sparsity across multiple layers. For a complex, multi-layered pre-trained model, it ensures that pruning is applied holistically, rather than layer by layer in isolation.
#     Optimized Resource Allocation: It minimizes the risk of over-pruning critical parts of the model by evaluating the importance of weights globally rather than locally. This allows the model to intelligently prune more from less sensitive layers and less from more sensitive layers, which is crucial for retaining performance on a pre-trained network.
#     Overall Performance Focus: This approach is beneficial when the overall model performance is more critical than the performance of individual layers. For pre-trained models used in transfer learning, maintaining high overall accuracy after compression is usually paramount.
#     Uniformly Sparse Models: Global pruning is particularly effective for creating uniformly sparse models that can adapt to various processing environments without heavily compromising accuracy. This makes the pruned pre-trained model more versatile for deployment.
# 
# While unstructured pruning can maximize sparsity, it relies on hardware support for efficiency, which may not always be available, especially for fine-grained sparsity. Structured pruning, while improving inference speed and reducing memory footprint by removing entire units, might be too aggressive if not carefully managed, potentially leading to a larger accuracy drop on complex pre-trained architectures compared to global pruning's more nuanced approach. Global pruning offers a balanced strategy for optimizing pre-trained models by considering the network's entire weight distribution, making it a highly relevant choice for practical deployment.
# =============================================================================
# %% Dataset and Training Configuration
# Configure DEVICE to use a CUDA-enabled GPU if available, otherwise it defaults to the CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# %% Load Dataset
# For the dataset, revisit the subset of Fruit and Vegetable Disease (Healthy vs Rotten). As a quick refresher on its statistics, this particular subset is structured with 28 classes, each corresponding to a subdirectory name and containing 200 images.
# Define the path to the image dataset.
dataset_path = "./fruit_and_vegetable_subset"

# %% Create data loaders for model training.
train_loader, val_loader = helper_utils.get_dataloaders(dataset_path)

# %% Set the number of classes.
# Get the number of classes dynamically from the train_loader's dataset
num_classes = len(train_loader.dataset.subset.dataset.classes)
print("Number of classes:", num_classes)

# %% Set Training Epochs
num_epochs = 2

# %% Unpruned Model Baseline
# =============================================================================
# Now that you have set up your dataset and training configurations, which will be identical for both the unpruned and pruned models, it's time to proceed with model training.
# 
# First, establish the unpruned model baseline, which will serve as the critical reference point for your comparative analysis.
# Loading and Training the Unpruned Model
# 
#     Initialize the ResNet18 model.
#         helper_utils.load_resnet18() loads the ResNet18 architecture with weights pre-trained on ImageNet from a local file.
#     helper_utils.replace_final_layer() adapts the model for transfer learning by replacing only the final fully connected layer to match the number of classes.
#         This function does not freeze any middle layers, ensuring all layers remain trainable. This is vital for conducting a fair comparison with the pruned model later, as frozen layers would not be affected by weight removal or subsequent fine-tuning during the pruning process.
# 
# =============================================================================
# Load the Resnet18 model with ImageNet weights
resnet18_model_unpruned = helper_utils.load_resnet18()

# Adapt the model for transfer learning with the new number of classes
model_unpruned = helper_utils.replace_final_layer(resnet18_model_unpruned, num_classes)

# %% This cell executes a standard training loop to fine-tune the unpruned model on your dataset.
# The loop will return the trained model and its performance metrics.
# These metrics include final Accuracy, Precision, Recall, and F1-Score.
# Train the unpruned model
trained_unpruned_model, unpruned_metrics = helper_utils.training_loop(model_unpruned,
                                                                      train_loader,
                                                                      val_loader,
                                                                      num_epochs,
                                                                      DEVICE,
                                                                      num_classes
                                                                     )

# %% Save the state dictionary and metrics of the trained unpruned model to file.
helper_utils.save_unpruned_model_and_metrics(trained_unpruned_model, unpruned_metrics)

# %% Print the key performance metrics of the unpruned model after training is complete.
print(f"""
-- After Training for {num_epochs} Epoch(s) --
Final Accuracy:           {unpruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {unpruned_metrics['precision']:.4f}
Final Recall (Macro):     {unpruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {unpruned_metrics['f1_score']:.4f}""")

# %% Configuring and Training the Pruned Model
# =============================================================================
# Now that you have successfully established the unpruned model baseline, it's time to explore the effects of pruning. Now, you will train a pruned version of the ResNet18 model, ensuring that all dataset and training configurations remain identical to those used for the unpruned baseline. This approach guarantees a fair and direct comparison of performance and model characteristics, allowing you to accurately assess the benefits of pruning.
# 
#     You'll start by re-initializing the Resnet18 model to ensure a clean slate.
#     Similar to the unpruned baseline model, replace the final classifier layer for transfer learning.
#         All of the middle layers are not frozen, allowing their weights to be modified by pruning and subsequent fine-tuning. This is essential for demonstrating the impact of pruning on a pre-trained network.
# =============================================================================
# Load the Resnet18 model with ImageNet weights
resnet18_model_pruned = helper_utils.load_resnet18()

# Adapt the model for transfer learning with the new number of classes
model_pruned = helper_utils.replace_final_layer(resnet18_model_pruned, num_classes)

# %%
# =============================================================================
# In the earlier sections with the SimpleModel, you manually checked things like model.named_parameters() and model.named_buffers() to see the _orig and _mask attributes. You also calculated sparsity for individual layers to verify the pruning.
# 
# But now, with a much larger model like ResNet18 and the intention to apply global pruning across many layers, checking each layer individually or manually calculating sparsity for every part will be quite cumbersome.
# 
# Instead, you'll use the analyze_model_sparsity function, which will provide a consolidated, overall view of the model's sparsity across all its weighted layers. Instead of going layer-by-layer, it will efficiently give you a single percentage that reflects how much of the entire network has been pruned. This is particularly relevant for global pruning, where the zeros are distributed across layers based on a global criterion, rather than a fixed amount per layer. It will provide a quick and quantitative assessment of the overall impact of pruning efforts on this larger, more complex architecture.
# 
#     Initialize counters for total and zero parameters.
#     Iterate through all named modules (layers) in the model.
#     For modules with a weight tensor, sums their elements to total_params and counts zero elements to total_zero_params.
#     Calculates overall sparsity as the percentage of zero parameters out of the total.
# =============================================================================



