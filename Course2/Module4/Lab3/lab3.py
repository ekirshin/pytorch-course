#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 M4 L3
Created on Tue Feb 24 10:25:24 2026

@author: ek
"""

import sys
import time
import warnings

# Redirect stderr to a black hole to catch other potential messages
class BlackHole:
    def write(self, message):
        pass
    def flush(self):
        pass
sys.stderr = BlackHole()

# Ignore Python-level UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# %% 
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms

import helper_utils

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)

# %%
class CIFAR10DataModule(pl.LightningDataModule):
    """A LightningDataModule for the CIFAR10 dataset."""

    def __init__(self, data_dir='./data', batch_size=128, num_workers=0):
        """
        Initializes the DataModule.

        Args:
            data_dir (str): Directory to store the data.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
        """
        # Call the constructor of the parent class (LightningDataModule).
        super().__init__()
        # Store the data directory path.
        self.data_dir = data_dir
        # Store the batch size for the DataLoaders.
        self.batch_size = batch_size
        # Store the number of worker processes for data loading.
        self.num_workers = num_workers
        # Define a sequence of transformations to be applied to the images.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if not already present."""
        
        # Download the training split of CIFAR10.
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        # Download the testing split of CIFAR10.
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Assigns train/val datasets for use in dataloaders.

        Args:
            stage (str, optional): The stage of training (e.g., 'fit', 'test').
                               The Lightning Trainer requires this argument, but it is not
                               utilized in this implementation as the setup logic is the
                               same for all stages. Defaults to None.
        """
        
        # Create the training dataset instance and apply the transformations.
        self.cifar_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
        # Create the validation dataset instance (using the test set) and apply transformations.
        self.cifar_val = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        # The DataLoader handles batching, shuffling, and parallel data loading.
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        # Shuffling is not necessary for the validation set.
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
# %%
class CIFAR10LightningModule(pl.LightningModule):
    """A flexible LightningModule for CIFAR10 image classification."""

    def __init__(self, 
                 learning_rate=1e-3, 
                 weight_decay=0.01,
                 conv_channels=(32, 64, 128),
                 linear_features=512,
                 num_classes=10):
        """
        Initializes the LightningModule with configurable layer parameters.

        Args:
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay (L2 penalty) for the optimizer.
            conv_channels: A tuple specifying the output channels for each
                           convolutional block.
            linear_features: The number of features in the hidden fully
                             connected layer.
            num_classes: The number of output classes for the classification task.
        """
        # Call the constructor of the parent class.
        super().__init__()
        # Save the hyperparameters passed to the constructor.
        self.save_hyperparameters()
        
        # Calculate the flattened size of the feature maps after the final
        # pooling layer. This is needed to define the input size of the
        # first fully connected layer.
        flattened_size = self.hparams.conv_channels[-1] * 4 * 4
        
        # Define the model's architecture using a sequential container.
        self.model = nn.Sequential(
            nn.Conv2d(3, self.hparams.conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hparams.conv_channels[0], self.hparams.conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hparams.conv_channels[1], self.hparams.conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(flattened_size, self.hparams.linear_features),
            nn.ReLU(),
            nn.Linear(self.hparams.linear_features, self.hparams.num_classes)
        )
        
        # Initialize the loss function.
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize metrics to track accuracy for training and validation.
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor containing a batch of images.

        Returns:
            The output tensor (logits) from the model.
        """
        # Pass the input through the sequential model.
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        Performs a single training step.
    
        Args:
            batch (Any): The data batch from the dataloader.
            batch_idx (int, optional): The index of the current batch. The Lightning Trainer
                                     requires this argument, but it is not utilized in this
                                     implementation. Defaults to None.
        """
        # Unpack the batch into inputs (images) and labels.
        inputs, labels = batch
        # Perform a forward pass to get the model's predictions (logits).
        outputs = self(inputs)
        # Calculate the loss.
        loss = self.loss_fn(outputs, labels)

        # Log the training loss
        self.log("train_loss", loss)
        
        # Return the loss to Lightning for backpropagation.
        return loss

    def validation_step(self, batch, batch_idx=None):
        """
        Performs a single validation step.
    
        Args:
            batch (Any): The data batch from the dataloader.
            batch_idx (int, optional): The index of the current batch. The Lightning Trainer
                                     requires this argument, but it is not utilized in this
                                     implementation. Defaults to None.
        """
        # Unpack the batch into inputs (images) and labels.
        inputs, labels = batch
        # Perform a forward pass to get the model's predictions (logits).
        outputs = self(inputs)
        # Calculate the loss.
        loss = self.loss_fn(outputs, labels)

        # Log the validation loss
        self.log("val_loss", loss, prog_bar=True)
        # Update the validation accuracy metric with the current batch's results.
        self.val_accuracy(outputs, labels)
        # Log the validation accuracy
        self.log("val_accuracy", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures and returns the model's optimizer.

        Returns:
            An instance of the optimizer.
        """
        # Create and return the AdamW optimizer.
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

#%% 
class PerformanceCallback(Callback):
    """
    A Lightning Callback to collect key performance metrics during the training
    and validation lifecycle.

    This callback measures:
    - Validation accuracy
    - Peak GPU memory usage during training
    """

    def __init__(self):
        """Initializes storage for the performance metrics to be collected."""
        # Initialize a list to store performance metrics
        self.metrics = []
        
        # Initialize an attribute to store peak memory usage
        self.peak_memory_mb = None

    def on_train_start(self, trainer, pl_module=None):
        """
        Resets peak GPU memory statistics at the start of training.
    
        Args:
            trainer (pl.Trainer): The main Lightning Trainer instance.
            pl_module (pl.LightningModule, optional): The LightningModule being trained.
                                                     Required by the callback hook but not
                                                     utilized here. Defaults to None.
        """
        # Check if CUDA is available for GPU operations
        if torch.cuda.is_available():
            # Reset the peak memory statistics on the root device
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
            # Clear the CUDA cache to free up memory
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module=None):
        """
        Aggregates and logs metrics at the end of a validation epoch.
    
        Args:
            trainer (pl.Trainer): The Lightning Trainer instance.
            pl_module (pl.LightningModule, optional): The LightningModule being validated.
                                                     Required by the callback hook but not
                                                     utilized here. Defaults to None.
        """
        # Check if the trainer is not in sanity checking mode
        if not trainer.sanity_checking:
            # Get the metrics collected by the trainer's callbacks
            metrics = trainer.callback_metrics
            # Append a dictionary with the validation accuracy to the metrics list
            self.metrics.append({
                "val_accuracy": metrics["val_accuracy"].item() * 100
            })

    def on_train_end(self, trainer, pl_module=None):
        """
        Records the peak GPU memory usage at the end of training.
    
        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule, optional): The LightningModule that was trained.
                                                     Required by the callback hook but not
                                                     utilized here. Defaults to None.
        """
        # Check if CUDA is available for GPU operations
        if torch.cuda.is_available():
            # Get the maximum memory allocated on the root device
            peak_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device)
            # Convert the peak memory from bytes to megabytes
            self.peak_memory_mb = peak_memory / 1024**2
# %%
def run_training(model, data_module, num_epochs, precision, grad_accum, performance_callback):
    """
    Configures and runs a Lightning training process.

    Args:
        model (pl.LightningModule): The model to be trained.
        data_module (pl.LightningDataModule): The data module that provides the datasets.
        num_epochs (int): The total number of epochs for training.
        precision (str): The numerical precision for training ('32-true', '16-mixed').
        grad_accum (int): The number of batches to accumulate gradients over.
        performance_callback (pl.Callback): A callback to measure performance metrics.

    Returns:
        pl.Trainer: The trainer instance after fitting is complete.
    """
    
    # Initialize a Lightning Trainer with specific parameters
    trainer = pl.Trainer(
        max_epochs=num_epochs,               # Set the maximum number of training epochs
        accelerator="auto",                  # Automatically select the best accelerator (e.g., CPU, GPU)
        devices=1,                           # Use a single device for training
        precision=precision,                 # Set the numerical precision for the training process
        accumulate_grad_batches=grad_accum,  # Configure the number of batches to accumulate gradients before updating weights
        callbacks=[performance_callback],    # Provide a list of callbacks to be used during training
        logger=False,                        # Disable logging for this training run
        enable_progress_bar=True,            # Enable the progress bar to show training progress
        enable_model_summary=False,          # Disable the model summary
        enable_checkpointing=False           # Disable model checkpointing
    )
    
    # Start the training process using the provided model and data module
    trainer.fit(model, data_module)
    
    # Return the configured and trained trainer instance
    return trainer            
# %%
def run_optimization(name, data_module, num_epochs, precision, grad_accum):
    """
    Orchestrates and runs a single, complete optimization experiment.

    Args:
        name (str): The display name for the experiment (e.g., "Mixed Precision").
        data_module (pl.LightningDataModule): The data module for training and validation.
        num_epochs (int): The number of epochs to train for.
        precision (str): The training precision to use ('32-true', '16-mixed').
        grad_accum (int): The number of batches for gradient accumulation.
        
    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
        1. The summarized results for the comparison table.
        2. The detailed data required for generating plots.
    """
    
    # Set up the experiment
    print(f"\n--- Running Experiment: {name} (For {num_epochs} Epochs) ---")
    
    model = CIFAR10LightningModule()
    performance_callback = PerformanceCallback()

    # Execute the training
    trainer = run_training(
        model=model,
        data_module=data_module,
        num_epochs=num_epochs,
        precision=precision,
        grad_accum=grad_accum,
        performance_callback=performance_callback,
    )

    # Process the results
    accuracies = [m["val_accuracy"] for m in performance_callback.metrics]
    
    current_results = {
        "optimization": name,
        "final_acc": accuracies[-1],
        "peak_mem_mb": performance_callback.peak_memory_mb,
    }
    
    plot_info = {**current_results, "accuracies": accuracies}
    
    # Return the findings
    return current_results, plot_info

# %%
# Set the number of epochs for each run
num_epochs = 5
# Define the number of samples to be processed in each batch
batch_size = 256
# Set the number of parallel processes for data loading
num_workers = 4

# Initialize an empty list to store the final summary results for the comparison table
results = []
# Initialize an empty list to store detailed data for generating plots
plot_data = []

# Create an instance of the DataModule, configuring it with the specified batch size and number of workers
data_module = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)
# %%
res, p_data = run_optimization(
    name="Standard",
    precision="32-true",
    grad_accum=1,
    data_module=data_module,
    num_epochs=num_epochs,
)

results.append(res)
plot_data.append(p_data)

# %% Mixed-precision
res, p_data = run_optimization(
    name="Mixed Precision",
    precision="16-mixed",
    grad_accum=1,
    data_module=data_module,
    num_epochs=num_epochs,
)

results.append(res)
plot_data.append(p_data)

# %% Gradient Accumulation: benefits of reducing the batch size, to reduce GPU memory without compromising your accuracy
# Data module setup
batch_size = 128
data_module = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)

# Gradient accumulation setup
effective_batch_size = 256
grad_accum = effective_batch_size // batch_size

res, p_data = run_optimization(
    name="Gradient Accumulation (Effective BS: 256-128)",
    precision="32-true",
    grad_accum=grad_accum,
    data_module=data_module,
    num_epochs=num_epochs,
)

results.append(res)
plot_data.append(p_data)

# %% Frad accumulation 
try:
    # Data module setup
    batch_size = 64
    data_module = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)

    res, p_data = run_optimization(
        name="Gradient Accumulation (Effective BS: 256-64)",

        # Set precision for gradient accumulation
        precision="32-true", ### Add your code here
        # Set the accumulation steps to achieve an effective batch size of 256
        grad_accum=effective_batch_size // batch_size, ### Add your code here

        data_module=data_module,
        num_epochs=num_epochs,
    )

    results.append(res)
    plot_data.append(p_data)

except Exception as e:
    print("\033[91mSomething went wrong, try again!")
    raise e


# %% Combined techniques
# Data module setup
batch_size = 128
data_module = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)

# Gradient accumulation setup
effective_batch_size = 256
grad_accum = effective_batch_size // batch_size

res, p_data = run_optimization(
    name="Combined (Effective BS: 256-128)",
    precision="16-mixed",
    grad_accum=grad_accum,
    data_module=data_module,
    num_epochs=num_epochs,
)

results.append(res)
plot_data.append(p_data)

# %%
try:
    # Data module setup
    batch_size = 64
    data_module = CIFAR10DataModule(batch_size=batch_size, num_workers=num_workers)
    grad_accum = effective_batch_size // batch_size

    res, p_data = run_optimization(
        name="Combined (Effective BS: 256-64)",

        # Use the setting for mixed precision
        precision="16-mixed", ### Add your code here
        # Set the accumulation steps to achieve an effective batch size of 256
        grad_accum=grad_accum, ### Add your code here    
        
        data_module=data_module,
        num_epochs=num_epochs,
    )

    results.append(res)
    plot_data.append(p_data)

except Exception as e:
    print("\033[91mSomething went wrong, try again!")
    raise e

# %%
# Display the comparison table of all experiment results
opt_results = helper_utils.optimization_results(results)
# %%
# Visualize the final accuracy to compare each optimization technique
helper_utils.plot_final_accuracy(results)
# %%
# Visualize the peak memory usage for each optimization
helper_utils.plot_peak_memory(results)


