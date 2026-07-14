# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:39:44 2026

C3 M4 L1. PyTorch Model Training with MLflow & Lightning: Tracking & Management.

Welcome to this hands-on lab! You've likely experienced the frustration of training a model for hours, achieving great accuracy, only to have it disappear when your notebook kernel restarts. This happens because a trained model exists only in memory; to preserve your work, you need a way to save it to disk, a process known as serialization.

This notebook moves you beyond simple training and into the essential practice of robust model management. You'll apply these concepts by training a PyTorch CNN using PyTorch Lightning and integrating a powerful tool, MLflow, to systematically track and organize your work, ensuring no effort is ever lost.

By the end of this lab, you'll have hands-on experience with:

    Using Lightning to simplify and organize your training code with LightningModule and LightningDataModule.

    Creating custom Lightning Callbacks to extend functionality during training.

    Setting up an MLflow experiment to serve as a dedicated container for your project's training runs.

    Logging hyperparameters and tracking metrics so you always have a record of the exact settings and performance of each run.

    Saving model checkpoints during training, creating snapshots of your best-performing models that you can reload instantly.

    Logging artifacts, such as your saved model files and performance plots, directly to MLflow for easy access and comparison.

    Accessing your results both through the interactive MLflow UI and programmatically with the Python client.


@author: ekirshin
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

import datetime
import logging
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy, ConfusionMatrix

import helper_utils

torch.set_float32_matmul_precision('medium')
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# %%  Random Seed Initialization
# Set a global random seed for PyTorch CPU and GPU operations.
# This is a crucial step for ensuring reproducibility of training results.

# Define the global random seed value
RANDOM_SEED = 42

# Set seed for PyTorch CPU operations
torch.manual_seed(RANDOM_SEED)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Set seed for PyTorch GPU operations on all available GPUs
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% CIFAR-10 DataModule
# Create a LightningDataModule to encapsulate all data loading logic including transformations, dataset downloads, and dataloader creation.
class CIFAR10DataModule(pl.LightningDataModule):
    """A LightningDataModule for the CIFAR10 dataset."""

    def __init__(self, data_dir='./CIFAR10_data', batch_size=128, num_workers=2):
        """
        Initializes the DataModule.

        Args:
            data_dir (str): Directory to store the data.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations for training data
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # Define transformations for validation data
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # CIFAR-10 class labels
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            print("CIFAR10 Data folder found locally. Loading from local.\n")
        else:
            print("CIFAR10 Data folder not found locally. Downloading data.\n")
            
        # Download the dataset (will skip if already exists)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Assigns train/val datasets for use in dataloaders.

        Args:
            stage (str, optional): The stage of training (e.g., 'fit', 'test').
        """
        # Create the training dataset
        self.cifar_train = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.transform_train
        )
        
        # Create the validation dataset
        self.cifar_val = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.transform_test
        )
    
    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return torch.utils.data.DataLoader(
            self.cifar_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return torch.utils.data.DataLoader(
            self.cifar_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

# %% # Instantiate the data module
data_module = CIFAR10DataModule()

# Get class names for later use
classes = data_module.classes

# %% CNN Model Definition as LightningModule
# Define a SimpleCNNLightning class, which extends pl.LightningModule and encapsulates the model architecture, training logic, and optimizer configuration.
class SimpleCNN(pl.LightningModule):
    """A Lightning-wrapped CNN for CIFAR10 image classification."""
    
    def __init__(self, learning_rate=0.001):
        """
        Initializes the LightningModule.
        
        Args:
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Define the model architecture
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10)
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x: The input tensor containing a batch of images.
            
        Returns:
            The output tensor (logits) from the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        
        Args:
            batch: The data batch from the dataloader.
            batch_idx: The index of the current batch.
            
        Returns:
            The loss value for backpropagation.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        
        Args:
            batch: The data batch from the dataloader.
            batch_idx: The index of the current batch.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        """
        Configures and returns the optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# %%  Confusion Matrix Plotting Function
# Define plot_confusion_matrix to generate and save a visual confusion matrix from a pre-computed matrix tensor.
# MLflow can log various artifacts, including graphical plots like confusion matrices, to provide deeper insights into model performance.
def plot_confusion_matrix(cm, class_names):
    """
    Generates and saves a graphical representation of a confusion matrix.

    Args:
        cm (numpy.ndarray): A square adjacency matrix representing the counts of 
                            true vs predicted classifications.
        class_names (list): A list of strings containing the names of the 
                            categories for the axes.

    Returns:
        output_filename (str): The file path where the confusion matrix image 
                               was saved.
    """
    # Initialize a new figure with specified dimensions
    plt.figure(figsize=(10, 8))
    # Display the matrix data as an image using a blue color map
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Set the title for the plot
    plt.title('Confusion Matrix')
    # Add a color bar scale to the side of the plot
    plt.colorbar()
    # Determine the positions for the axis ticks
    tick_marks = np.arange(len(class_names))
    # Apply class names to the x-axis with a 45-degree rotation
    plt.xticks(tick_marks, class_names, rotation=45)
    # Apply class names to the y-axis
    plt.yticks(tick_marks, class_names)

    # Calculate a threshold for determining text contrast
    thresh = cm.max() / 2.
    # Iterate through the rows of the matrix
    for i in range(cm.shape[0]):
        # Iterate through the columns of the matrix
        for j in range(cm.shape[1]):
            # Place the numeric value text within the specific cell
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     # Toggle text color between white and black for readability
                     color="white" if cm[i, j] > thresh else "black")

    # Automatically adjust subplot parameters for a clean layout
    plt.tight_layout()
    # Assign the label for the vertical axis
    plt.ylabel('True label')
    # Assign the label for the horizontal axis
    plt.xlabel('Predicted label')

    # Assign a string for the target output file
    output_filename = 'confusion_matrix.png'
    # Export the current figure to the local file system
    plt.savefig(output_filename)
    # Release the memory associated with the plot figure
    plt.close()

    # Return the path of the saved image file
    return output_filename

# %% MLflow Logging Callback
# =============================================================================
# While Lightning provides a built-in MLFlowLogger for standard experiment tracking, this lab uses a custom Callback. This approach allows for more advanced control, such as logging custom-generated plots and implementing unique checkpointing logic that the built-in logger does not support.
# Define a custom Lightning Callback to automatically log metrics, hyperparameters, and artifacts to MLflow during training.
# =============================================================================
class MLflowLoggingCallback(Callback):
    """
    A callback module for integrating MLflow tracking with the training lifecycle.
    
    Args:
        classes (tuple): A collection of labels used for mapping indices to 
                         human-readable class names in the confusion matrix.
    """
    
    def __init__(self, classes):
        """
        Initializes the callback with experiment settings and local storage.
        
        Args:
            classes (tuple): The sequence of class names for visualization.
        """
        # Execute the initializer of the parent Callback class
        super().__init__()
        # Store the class labels for later use in artifact generation
        self.classes = classes
        # Initialize a tracker for the highest recorded accuracy
        self.best_accuracy = 0
        # Define the local directory path for storing the best model state
        self.model_save_dir = "./best_model"
        # Ensure the model storage directory exists on the file system
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def on_train_start(self, trainer, pl_module):
        """
        Records experiment parameters at the beginning of the training process.
        
        Args:
            trainer (pytorch_lightning.Trainer): The current trainer instance.
            pl_module (pytorch_lightning.LightningModule): The model being trained.
        """
        # Log the architecture type of the model
        mlflow.log_param("model_type", "SimpleCNN")
        # Log the specific optimizer algorithm being utilized
        mlflow.log_param("optimizer", "Adam")
        # Log the starting learning rate from the model hyper-parameters
        mlflow.log_param("initial_lr", pl_module.hparams.learning_rate)
        # Log the name of the learning rate adjustment strategy
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        # Log the batch size configured in the data module
        mlflow.log_param("batch_size", trainer.datamodule.batch_size)
        # Log the global seed used for reproducibility
        mlflow.log_param("random_seed", RANDOM_SEED)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Logs performance metrics and manages checkpointing at the end of validation.
        
        Args:
            trainer (pytorch_lightning.Trainer): The current trainer instance.
            pl_module (pytorch_lightning.LightningModule): The model being validated.
        """
        # Avoid logging during the initial trainer sanity check phase
        if trainer.sanity_checking:
            return
        
        # Access the current dictionary of calculated metrics
        metrics = trainer.callback_metrics
        # Determine the current epoch index from the trainer
        current_epoch = trainer.current_epoch
        
        # Check for training loss in metrics and record it if present
        if "train_loss" in metrics:
            mlflow.log_metric("train_loss", metrics["train_loss"].item(), step=current_epoch)
        # Check for validation loss in metrics and record it if present
        if "val_loss" in metrics:
            mlflow.log_metric("val_loss", metrics["val_loss"].item(), step=current_epoch)
            
        # Extract the current learning rate from the primary optimizer
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        # Record the learning rate as a metric for the current step
        mlflow.log_metric("learning_rate", current_lr, step=current_epoch)
        
        # Process validation accuracy and handle model persistence
        if "val_acc" in metrics:
            # Convert the accuracy fraction to a percentage
            accuracy = metrics["val_acc"].item() * 100
            # Log the percentage accuracy to the experiment tracker
            mlflow.log_metric("accuracy", accuracy, step=current_epoch)
            
            # Evaluate if the current accuracy exceeds the historical best
            if accuracy > self.best_accuracy:
                # Update the best accuracy record
                self.best_accuracy = accuracy
                
                # Retrieve validation loss or use infinity if unavailable
                safe_val_loss = metrics.get("val_loss", torch.tensor(float('inf'))).item()
                
                # Construct the state dictionary for the checkpoint
                checkpoint = {
                    'epoch': current_epoch + 1,
                    'model_state_dict': pl_module.state_dict(),
                    'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                    'val_loss': safe_val_loss,
                    'accuracy': accuracy,
                    'random_seed': RANDOM_SEED 
                }
                
                # Generate a unique filename for the best model checkpoint
                checkpoint_filename = f'best_model_checkpoint_epoch_{current_epoch + 1}.pt'
                # Combine the directory and filename into a full path
                checkpoint_path = os.path.join(self.model_save_dir, checkpoint_filename)
                # Serialize and save the checkpoint to disk
                torch.save(checkpoint, checkpoint_path)
                
                # Upload the checkpoint file as an MLflow artifact
                mlflow.log_artifact(checkpoint_path)
    
    def on_train_end(self, trainer, pl_module):
        """
        Performs final logging of artifacts and the serialized model.
        
        Args:
            trainer (pytorch_lightning.Trainer): The current trainer instance.
            pl_module (pytorch_lightning.LightningModule): The trained model.
        """
        # Log the highest accuracy achieved throughout the entire run
        mlflow.log_metric("best_accuracy", self.best_accuracy)
        
        # Provide console feedback for the final artifact generation
        print("\nCalculating final confusion matrix for artifact logging...")
        # Initialize the confusion matrix metric on the appropriate device
        confmat_metric = ConfusionMatrix(task="multiclass", num_classes=10).to(pl_module.device)
        
        # Switch the model to evaluation mode
        pl_module.eval()
        # Disable gradient computation for inference
        with torch.no_grad():
            # Iterate through the validation data to collect predictions
            for batch in trainer.val_dataloaders:
                # Unpack images and labels from the current batch
                images, labels = batch
                # Transfer data to the same device as the model
                images, labels = images.to(pl_module.device), labels.to(pl_module.device)
                # Obtain model logits
                outputs = pl_module(images)
                # Convert logits to class predictions
                preds = torch.argmax(outputs, dim=1)
                # Update the state of the confusion matrix metric
                confmat_metric.update(preds, labels)
        
        # Calculate the final matrix and move it to system memory
        final_cm = confmat_metric.compute().cpu().numpy()
        # Generate the visualization and retrieve the file path
        cm_path = plot_confusion_matrix(final_cm, self.classes)
        # Log the confusion matrix image to the experiment
        mlflow.log_artifact(cm_path)
        
        # Extract a sample input for model signature generation
        input_example_tensor, _ = next(iter(trainer.val_dataloaders))
        # Convert the sample tensor to a numpy array
        input_example_numpy = input_example_tensor.cpu().numpy()
        
        # Relocate the model to the CPU for standardized serialization
        pl_module.to("cpu")
        
        # Register the trained PyTorch model within the MLflow registry
        mlflow.pytorch.log_model(
            pytorch_model=pl_module,
            artifact_path="cifar10_cnn_model_final",
            input_example=input_example_numpy
        )
        
        # Output the final status to the console
        print(f'\nFinished Training. Best accuracy: {self.best_accuracy:.2f}%')
        
# %% MLflow Experiment Setup
# =============================================================================
# Set the active MLflow experiment to "CIFAR10_CNN" using mlflow.set_experiment().
# 
#     By default, the name of the experiment is set to CIFAR10_CNN. You can rename it to anything you want.
# 
# Organizing runs within named experiments is a fundamental aspect of using MLflow for structured experiment tracking.        
# =============================================================================
# Set or create an MLflow experiment named "CIFAR10_CNN"
mlflow.set_experiment("CIFAR10_CNN")

# %% MLflow Run and Training with Lightning Trainer
# =============================================================================
# Begin an MLflow run context using mlflow.start_run(), which groups all subsequent logging under a single run ID.
# Lightning Trainer Setup:
# 
#     Configure the Lightning Trainer with essential parameters:
#         max_epochs: Number of training epochs
#         accelerator="auto": Automatically detect and use available hardware (GPU/CPU)
#         devices=1: Use a single device
#         logger=False: Disable default Lightning logging (we use MLflow instead)
#         callbacks: Pass the MLflowLoggingCallback to handle all MLflow logging
#         enable_progress_bar=True: Show training progress
# 
# Training Execution:
# 
#     Call trainer.fit(model, data_module) to run the entire training loop automatically
#     Lightning handles all the training/validation loops, gradient updates, and device management
#     The callback handles all MLflow logging throughout the process
# 
# 
# =============================================================================
# Start an MLflow run context
with mlflow.start_run() as run:
    
    # Define the total number of training epochs
    num_epochs = 10
    
    # Create the model
    model = SimpleCNN(learning_rate=0.001)
    
    # Create the MLflow logging callback
    mlflow_callback = MLflowLoggingCallback(classes=classes)
    
    # Configure the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        logger=False,  # Disable default Lightning logging
        callbacks=[mlflow_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=False  # We handle checkpointing in the callback
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    print(f'\nMLflow run id: {run.info.run_id}')
    print('To view results run mlflow ui')
    
# %% MLflow UI Setup
# =============================================================================
# Navigating the UI
# 
#     The helper_utils.show_ui_navigation_instructions() function will render a step-by-step guide on how to use the MLflow UI.
#         To show the instructions, pass display_instructions=True.
#         To hide the instructions, you can omit the argument, as it defaults to False.
# =============================================================================
# Set the parameter to True to see the instructions
helper_utils.show_ui_navigation_instructions(display_instructions=True)    

# %% Terminating the MLflow UI Server
# When you are done, you can stop the server by running mlflow_process.terminate() in a notebook cell.
# %% Running the MLflow UI
# =============================================================================
# Use start_mlflow_ui() to launch the MLflow UI server with nginx reverse proxy support.
# The server will run in the background, allowing you to continue with the notebook without interruption.
# =============================================================================
# Start the MLflow UI server (runs in background)
mlflow_process = helper_utils.start_mlflow_ui()

# %% Alternative to UI: Viewing MLflow Logs via Python Client in Notebook
# =============================================================================
# While the MLflow UI provides a comprehensive graphical interface for exploring your experiment runs, there are times when accessing and reviewing your MLflow logs directly within your Jupyter Notebook can be more convenient or suitable for specific workflows. This section demonstrates how you can use MLflow's Python client to programmatically retrieve and display information about your experiment runs without leaving the notebook environment.
# 
# The approach shown here offers a minimalistic view, primarily focusing on fetching details for specific runs based on their Run IDs. However, this is just a starting point. The MLflow Python client is quite powerful, and you can easily adapt and expand upon these methods to customize the information you want to extract and the way it's presented, tailoring it to your specific analytical or reporting needs.
# Initialize MLflow Client to interact with MLflow.
# Search and display all experiments and, for each, list their runs with start times and names, prompting you to copy a run_id.
# =============================================================================
client = mlflow.tracking.MlflowClient()

print("MLflow Client Initialized.\n")
print("Listing all Run IDs from all experiments (descending by start time)...\n")

experiments = client.search_experiments()

if not experiments:
    print("No experiments found.")
else:
    all_runs_listed = False
    for exp in experiments:
        print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"])
        
        if not runs:
            print("  No runs found in this experiment.")
        else:
            all_runs_listed = True
            for run_info in runs:
                start_time_seconds = run_info.info.start_time / 1000.0
                start_time_formatted = datetime.datetime.fromtimestamp(start_time_seconds).strftime('%Y-%m-%d %H:%M:%S')
                
                run_name_tag = run_info.data.tags.get('mlflow.runName')
                display_name = f" (Name: {run_name_tag})" if run_name_tag else ""
                print(f"  Run ID: {run_info.info.run_id}{display_name} - Started: {start_time_formatted}")
        print("-" * 30)
    if not all_runs_listed:
        print("\nNo runs found across any experiment.")

print("\nCopy the Run ID you want to inspect from the list above.")

# %% # Paste the Run ID you copied into the run_id variable below.
run_id = "5f2fba92dfb04356b43401f235234aa8"  # <-- PASTE YOUR RUN ID BETWEEN THE QUOTES

# %%
# =============================================================================
# Use display_mlflow_run_details function to fetch and display a formatted summary of the run, which includes:
# 
#     The selected Run ID and the name/ID of the experiment it belongs to.
#     The values of key hyperparameters that were logged for the run.
#     The final values of all tracked metrics.
#     A list of all artifacts associated with the run.
# =============================================================================
try:
    # Validate the run_id and print a confirmation.
    if not run_id or not run_id.strip():
        print("Error: Please paste a valid Run ID. It cannot be empty or just whitespace.")
    else:
        print(f"Run ID set to: {run_id}")
        # Call the helper function to display the run details.
        helper_utils.display_mlflow_run_details(run_id)

except NameError:
    print("Error: The 'run_id' variable is not defined. Please make sure it's set in the try block.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% Conclusion

# =============================================================================
# Congratulations on completing this lab! You have successfully trained a CNN using PyTorch Lightning and, more importantly, implemented a systematic workflow to manage and preserve your results.
# 
# You've put foundational MLOps principles into practice with modern deep learning tools. By using PyTorch Lightning, you've seen how to organize your code into clean, reusable components (LightningModule and LightningDataModule) and how to extend functionality with custom Callbacks. This approach dramatically simplifies your training code while maintaining full flexibility.
# 
# You implemented checkpointing to not only save your final model but also the best-performing version during training, complete with its optimizer state. By logging this checkpoint and other artifacts like the confusion matrix through your custom MLflow callback, you've moved past the chaos of managing countless model files with confusing names.
# 
# By combining Lightning with MLflow, you transformed your training process from a temporary session into a series of well-documented, reproducible experiments. Every hyperparameter, metric, and model file is now neatly organized and ready for comparison. The skills you've developed here in modern deep learning frameworks, serialization, and experiment tracking are fundamental for preparing any model for real-world deployment. You are now well-equipped to build more reliable and structured machine learning systems.
# =============================================================================

