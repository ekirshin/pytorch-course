# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:39:32 2026

@author: ekirshin
"""
import sys
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

#%% 
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms

import helper_utils

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)

#%% 
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
    
#%% 
class CIFAR10LightningModule(pl.LightningModule):
    """A flexible LightningModule for CIFAR10 image classification."""

    def __init__(self,
                 learning_rate=1e-3,
                 weight_decay=0.01,
                 conv_channels=(256, 512, 1024),
                 linear_features=2048,
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
        # Save the hyperparameters passed to the constructor. This makes them
        # accessible via `self.hparams` and logs them automatically.
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
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
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

        # Log the training loss at the end of each epoch.
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # Update the training accuracy metric with the current batch's results.
        self.train_accuracy(outputs, labels)
        # Log the training accuracy at the end of each epoch.
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
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

        # Log the validation loss at the end of each epoch.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Update the validation accuracy metric with the current batch's results.
        self.val_accuracy(outputs, labels)
        # Log the validation accuracy at the end of each epoch.
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures and returns the model's optimizer.

        Returns:
            An instance of the optimizer.
        """
        # Create and return the AdamW optimizer.
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

#%% # Instantiate the DataModule (2 workers).
dm_loader = CIFAR10DataModule(num_workers=2)

# Create an instance of the LightningModule.
model_baseline = CIFAR10LightningModule()

#%% A Quick Training Run
baseline_results = helper_utils.run_full_training(model_baseline, dm_loader)

print("\nTraining Complete!\n")
print("Final Training Metrics:")

print(f"\tTraining Accuracy:    {baseline_results['train_accuracy']}%")
print(f"\tValidation Accuracy:  {baseline_results['val_accuracy']}%")

#%% 
log_dir = "./profiler_output"

#%% # Configure the PyTorch Profiler
profiler = PyTorchProfiler(
    # Set the directory to save the profiler report
    dirpath=log_dir,
    # Specify the filename for the report
    filename="profile_report",
    # Define the profiling schedule (wait -> warmup -> active)
    # Total 14 steps
    schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
    # Enable memory usage profiling
    profile_memory=True
)

#%% # Initialize the Trainer
trainer = pl.Trainer(
    # Attach the configured profiler
    profiler=profiler,
    # Limit training to 14 steps to match the profiler's schedule
    max_steps=14,
    # Automatically select the hardware accelerator (e.g., GPU, CPU)
    accelerator="auto",
    # Use a single device for training
    devices=1,
    # Disable the default logger for a cleaner output
    logger=False,
    # Disable the model summary for the same reason
    enable_model_summary=False,
    # Disable automatic checkpointing
    enable_checkpointing=False
)

#%% # Start the training and profiling run.
trainer.fit(model_baseline, dm_loader)

# Print a confirmation message when done.
print("\nProfiling Complete!\n")

#%% # Display the top 10 most time-consuming operations from the profiler's report
top10_display = helper_utils.display_profiler_logs(profiler, head=10)

#%% # Display a focused summary of the profiler report for the baseline run.
# This filters for the overall time and the top 4 computational operations.
top10_model = helper_utils.display_model_computation_logs(profiler)

#%% Profiling a More Efficient Model
#%% Step 1: Configure a New Profiler
try:
    # Configure the PyTorch Profiler for the efficient model run
    profiler_efficient = PyTorchProfiler( ### Add your code here
        
        # Set the directory to save the profiler report
        dirpath=log_dir, ### Add your code here
        
        # Specify a new filename for the report
        filename="profile_report_efficient", ### Add your code here
        
        # Define the profiling schedule (wait -> warmup -> active)
        schedule=schedule(wait=2, warmup=2, active=10, repeat=1), ### Add your code here
        
        # Enable memory usage profiling
        profile_memory=True ### Add your code here
    )

    print("\033[92mPyTorchProfiler configured successfully!")

except Exception as e:
    print("\033[91mSomething went wrong, try again!")
    raise e

#%% Step 2: Configure a New Trainer
try:
    # Initialize the Trainer
    trainer_efficient = pl.Trainer( ### Add your code here
        
        # Attach the configured profiler
        profiler=profiler_efficient, ### Add your code here
        
        # Limit training to 14 steps to match the profiler's schedule
        max_steps=14, ### Add your code here
        
        # Automatically select the hardware accelerator (e.g., GPU, CPU)
        accelerator="auto", ### Add your code here
        
        # Use a single device for training
        devices=1, ### Add your code here
        
        # Disable the default logger for a cleaner output
        logger=False, ### Add your code here
        
        # Disable the model summary for the same reason
        enable_model_summary=False, ### Add your code here

        # Disable automatic checkpointing
        enable_checkpointing=False ### Add your code here
    )

    print("\033[92mTrainer configured successfully!")

except Exception as e:
    print("\033[91mSomething went wrong, try again!")
    raise e
    
# %% Step 3: Profile the Efficient Model
# Create a new instance of the model with a much simpler architecture.
model_efficient = CIFAR10LightningModule(
    conv_channels=(32, 64, 128),
    linear_features=512
)

# %%
try:
    # Start the second diagnostic run with the new, streamlined model.
    trainer_efficient.fit(model_efficient, dm_loader) ### Add your code here
    
    print("\nProfiling Complete!\n")
    
except Exception as e:
    print("\033[91mSomething went wrong, try again!")
    raise e

# %% Step 4: Analyzing the Efficient Profiler Results
# Display the top 10 most time-consuming operations from the profiler_efficient's report
top10_display_efficient = helper_utils.display_profiler_logs(profiler_efficient, head=10)    

# %% # Generate the comparison report.
comparison_df = helper_utils.display_comparison_report(profiler, profiler_efficient)

# %% Inspect metrics of the final efficient model
efficient_results = helper_utils.run_full_training(model_efficient, dm_loader)

print("\nTraining Complete!\n")
print("Final Training Metrics:")

print(f"\tTraining Accuracy:    {efficient_results['train_accuracy']}%")
print(f"\tValidation Accuracy:  {efficient_results['val_accuracy']}%")

# %% 
comparison_perf_df = helper_utils.display_metrics_comparison(baseline_results, efficient_results)
