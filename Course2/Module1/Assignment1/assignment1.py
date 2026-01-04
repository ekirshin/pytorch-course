#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 09:32:56 2025

@author: ek
"""

from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy, Precision, Recall

from helper_utils import evaluate_model, extract_attr, get_data_loaders, training_epoch

import helper_utils
#import unittests

#%%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

#%% Explore datasets
AIvsReal_path = "./AIvsReal_sampled"  

# Display example images
for split in ['train', 'test']:
    for category in ['real', 'fake']:
        helper_utils.show_random_images(split, category, AIvsReal_path, num_images=5)

#%% # GRADED CLASS: FlexibleCNN

class FlexibleCNN(nn.Module):
    """
    A customizable convolutional neural network (CNN) for image classification.
    It dynamically constructs convolutional blocks based on provided hyperparameters
    such as the number of layers, filter sizes, kernel sizes, and dropout rates.
    """

    def __init__(
        self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes=2
    ):
        """
        Initializes the FlexibleCNN.

        Args:
            n_layers (int): Number of convolutional layers.
            n_filters (list): Number of filters for each convolutional layer.
            kernel_sizes (list): Kernel sizes for each convolutional layer.
            dropout_rate (float): Dropout rate for regularization.
            fc_size (int): Number of units in the fully connected layer.
            num_classes (int): Number of output classes.
        """
        super(FlexibleCNN, self).__init__()

        self.num_classes = num_classes
        
        self.features = nn.ModuleList()
        in_channels = 3  # RGB input images

        ### START CODE HERE ###
        
        for i in range(n_layers): 
            # Create convolutional layer with dynamic parameters
            
            # Extract the number of filters and kernel size for the current layer, from n_filters and kernel_sizes
            out_channels = n_filters[i] 
            kernel_size = kernel_sizes[i] 
            
            
            padding = (kernel_size - 1) // 2 

            # Create a convolutional block, by using a `nn.Sequential` container to group layers together
            block = nn.Sequential(
                # Add a Convolutional layer `Conv2d` with parameters: `in_channels`, `out_channels`, `kernel_size`, and `padding`
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                # Add a Batch normalization layer `BatchNorm2d` with `num_features` as `out_channels` 
                nn.BatchNorm2d(num_features=out_channels),
                # Add a ReLU activation
                nn.ReLU(), 
                # Add a MaxPool2d layer `MaxPool2d`, with `kernel_size=2` and `stride=2`
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
                
            # Append the convolutional block to the features ModuleList
            self.features.append(block)

            # Update in_channels for the next layer (the input channels for the next layer is the output channels of the current layer)
            in_channels = out_channels
            

        ### END CODE HERE ###

        self.dropout_rate = dropout_rate
        self.fc_size = fc_size        
                
        # Classifier will be initialized after calculating flattened size
        self.classifier = None  
        self._flattened_size = None 
        

    def _create_classifier(self, flattened_size):
        """
        Creates the fully connected classifier part of the model based on the flattened feature size.

        Args:
            flattened_size (int): Size of the flattened feature maps.
        """
        ### START CODE HERE ###

        # Create the classifier using a Sequential container
        self.classifier = nn.Sequential(
            # Add a dropout layer with the dropout rate defined at initialization
            nn.Dropout(p=self.dropout_rate), 
            # Add a fully connected layer `Linear` with `in_features=flattened_size` and `out_features` as `fc_size`
            nn.Linear(in_features=flattened_size, out_features=self.fc_size),
            # Activation function
            nn.ReLU(), 
            # # Another dropout layer
            nn.Dropout(p=self.dropout_rate), 
            # Add the final fully connected layer with `in_features` as `fc_size` and `out_features` as `num_classes`
            nn.Linear(in_features=self.fc_size, out_features=self.num_classes),
        ) 

        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass of the FlexibleCNN.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output tensor (classification scores).
        """
        # Apply convolutional feature extraction layers
        for layer in self.features:
            x = layer(x)

        ### START CODE HERE ###

        # Flatten the output x for the classifier (start_dim=1 to keep the batch dimension)
        x = torch.flatten(x, 1)

        # Dynamically create classifier if it doesn't exist
        if self.classifier is None:
            # Get the size of the flattened feature maps from the x tensor
            self._flattened_size = x.size(1)

            # Create the classifier with the `_flattened_size` 
            self._create_classifier(self._flattened_size)

        ### END CODE HERE ###
            
            # Extract the device from the input tensor
            device = x.device
            
            # Move the classifier to the same device as the input tensor, to ensure compatibility with GPU/CPU
            if self.classifier is not None:
                self.classifier.to(device)

        # Classification
        return self.classifier(x)


#%% # Create the model with specific parameters
n_layers = 3
n_filters = [16, 32, 64]
kernel_sizes = [3, 3, 3]
dropout_rate = 0.5
fc_size = 128

model = FlexibleCNN(
    n_layers=n_layers,
    n_filters=n_filters,
    kernel_sizes=kernel_sizes,
    dropout_rate=dropout_rate,
    fc_size=fc_size,
).to(DEVICE)

resolution = 32
x_sample = torch.randn(1, 3, resolution, resolution).to(DEVICE)  # Example input tensor


# Forward pass through the model
output = model(x_sample)

# print the model features architecture
print(f"FlexibleCNN features architecture:\n{model.features}")

print(f"FlexibleCNN classifier architecture:\n{model.classifier}")

#%% Designing the Hyperparameter Search Space
# GRADED FUNCTION: design_search_space

def design_search_space(trial):
    """
    Design the search space for hyperparameter optimization of the FlexibleCNN model.
    This function uses Optuna to suggest hyperparameters for the CNN architecture and training process.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
    Returns:
        dict: A dictionary containing the suggested hyperparameters.    
    """
    # CNN Architecture Hyperparameters

    ### START CODE HERE ###
    
    # Use trial.suggest_* to set n_layers. Name it "n_layers", and set it to be an integer between 1 and 3.
    n_layers = trial.suggest_int("n_layers", 1, 3)
       
    # Use trial.suggest_* to set each filter size in n_filters.
    # Name each filter size as "n_filters_layer{i}" where i is the layer index and set it to be an integer between 8 and 64 with step 8.
    n_filters = [ 
        trial.suggest_int(f"n_filters_layer{i}", 8, 64, step=8) for i in range(n_layers)
    ] 
    
    # Use trial.suggest_* to set each kernel size in kernel_sizes.
    # Name each kernel size as "kernel_size_layer{i}" where i is the layer index and set it to be an integer between 3 and 5 with step 2.
    kernel_sizes = [ 
        trial.suggest_int(f"kernel_size_layer{i}", 3, 5, step=2) for i in range(n_layers)
    ] 

    # Use trial.suggest_* to set dropout_rate, name it "dropout_rate", and set it to be a float between 0.1 and 0.5.
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Use trial.suggest_* to set fc_size, name it "fc_size", and set it to be an integer between 64 and 512 with step 64.
    fc_size = trial.suggest_int("fc_size", 64, 512, step=64)

    
    # Training Hyperparameters

    # Use trial.suggest_* to set learning_rate, name it "learning_rate", and set it to be a float between 1e-4 and 1e-2 with logarithmic scale (log=True).
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Use trial.suggest_* to set resolution, name it "resolution", and set it to be one of [16, 32, 64].
    resolution = trial.suggest_categorical("resolution", [16, 32, 64])
    
    # Use trial.suggest_* to set batch_size, name it "batch_size", and set it to be one of [8, 16].
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    
    ### END CODE HERE ###

    return {
        "n_layers": n_layers,
        "n_filters": n_filters,
        "kernel_sizes": kernel_sizes,
        "dropout_rate": dropout_rate,
        "fc_size": fc_size,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "batch_size": batch_size,
    }

#%% # To verify that the design_search_space function works correctly, you can run it with a fixed trial.

# Create a fixed trial with specific hyperparameters
fixed_params = {
    "n_layers": 2,
    "n_filters_layer0": 16,
    "n_filters_layer1": 32,
    "kernel_size_layer0": 3,
    "kernel_size_layer1": 5,
    "dropout_rate": 0.001,
    "fc_size": 128,
    "learning_rate": 1e-3,
    "resolution": 32,
    "batch_size": 16,
}
toy_trial = optuna.trial.FixedTrial(fixed_params)

# Display the design search space for the fixed trial
pprint(design_search_space(toy_trial))

#%% 
# GRADED FUNCTION: objective_function

def objective_function(trial, device, dataset_path, n_epochs=4, silent=False, test=False):
    """
    Objective function for Optuna to optimize the hyperparameters of the FlexibleCNN model.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
        n_epochs (int): Number of epochs for training the model.
        silent (bool): If True, suppresses output during training and evaluation.
        test (bool): If True, extracts attributes from the trial for evaluation purposes.
    Returns:
        float: The accuracy of the model on the validation set.
    """

    # === construction of model, dataloaders ===
    
    ### START CODE HERE ###
    
    # use design_search_space to get the parameters for the trial
    params = design_search_space(trial)

    # add the transform to resize the images to the specified resolution in params
    transform = transforms.Compose([
            transforms.Resize((params["resolution"], params["resolution"])),
            transforms.ToTensor(), 
        ])

    # define the model using the FlexibleCNN class with the parameters from the trial
    model = FlexibleCNN(
        n_layers = params["n_layers"],
        n_filters = params["n_filters"],
        kernel_sizes = params["kernel_sizes"],
        dropout_rate = params["dropout_rate"],
        fc_size = params["fc_size"]
    ) 
    
    ### END CODE HERE ###

    # Initialize the dynamic classifier layer by passing a dummy input through the model
    # This ensures all parameters are instantiated before the optimizer is defined
    dummy_input = torch.randn(1, 3, params["resolution"], params["resolution"]).to(device)
    model = model.to(device)
    model(dummy_input)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(transform, params["batch_size"], dataset_path)
    
    # === Optimizer and Loss Function ===
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    loss_fcn = nn.CrossEntropyLoss()

    # === Training the model ===

    model = model.to(device)
    
    # = Training =
    for epoch in range(n_epochs):
        _ = training_epoch(
            model,
            train_loader,
            optimizer,
            loss_fcn,
            device,
            epoch,
            n_epochs,
            silent=silent,
        )

    # === Evaluation ===

    accuracy = evaluate_model(model, val_loader, device, silent=silent)

    # NOTE: the following line is only for evaluation purposes
    if test:
        extract_attr(trial, transform, model, params) 
    return accuracy

#%% 
# It takes about 1 minute to train for 1 epoch
# Run the objective function with a fixed trial 

fixed_trial = optuna.trial.FixedTrial({
    "n_layers": 2,
    "n_filters_layer0": 16,
    "n_filters_layer1": 32,
    "kernel_size_layer0": 3,
    "kernel_size_layer1": 3,
    "dropout_rate": 0.3,
    "fc_size": 128,
    "learning_rate": 0.001,
    "resolution": 32,
    "batch_size": 16,
})

#_ = helper_utils.run_silent_function(fixed_trial, objective_function)
objective_function(trial=fixed_trial, device=DEVICE, n_epochs=1, dataset_path=AIvsReal_path, silent=False, test=True) 

print('\n Some objects from the trial: \n')

print('transform:', fixed_trial.user_attrs['transform'])
print('\n model:', fixed_trial.user_attrs['model'])

#%% Navigating the Hyperparameter Landscape: Running the Optuna Study
storage = "sqlite:///example.db"
study_name = "AIvsReal_optimization"

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage)

#%% Create new study instead of loading pre-existing study 
study_name = "AIvsReal_optimization"
study = optuna.create_study(study_name=study_name)

#%% # continue with the study, with 2 more trials (about 9 minutes to run)
n_epochs = 3
study.optimize(lambda trial: objective_function(trial, n_epochs=n_epochs, device=DEVICE, dataset_path=AIvsReal_path), n_trials=2)

#%% 
# check the k best trials
df_trials = study.trials_dataframe()

# Sort the trials by value (accuracy) in descending order
df_trials.sort_values(by="value", ascending=False, inplace=True)

df_trials

#%%
k = 5
# Get the top k trials
best_k_trials = df_trials.head(k)
best_k_trials

#%% Analyzing efficiency
# GRADED FUNCTION: get_trainable_params

def get_trainable_params(model):
    """
    Calculate the total number of trainable parameters in the model.
    Args:
        model (nn.Module): The PyTorch model.
    Returns:
        int: Total number of trainable parameters in the model.
    """
    total_trainable_params = 0

    ### START CODE HERE ###

    # Get the model parameters
    model_parameters = model.parameters()
    # Iterate through the model parameters
    for param in model_parameters:
        # check if the parameter requires gradient
        if param.requires_grad:
            # Add the number of elements in the parameter to the total
            total_trainable_params += param.numel()
            
    ### END CODE HERE ###
    return total_trainable_params

#%%
n_layers = 3
n_filters = [16, 32, 64]
kernel_sizes = [3, 3, 3]
dropout_rate = 0.5
fc_size = 128

model = FlexibleCNN(
    n_layers=n_layers,
    n_filters=n_filters,
    kernel_sizes=kernel_sizes,
    dropout_rate=dropout_rate,
    fc_size=fc_size,
)

# Run a dummy pass to create the classifier layers
# otherwise get_trainable_params will only count the feature extractor
resolution = 32
dummy_input = torch.randn(1, 3, resolution, resolution)
model(dummy_input)

print("Total trainable parameters:", get_trainable_params(model))

#%% Add efficiency to the Optuna study results
def add_efficiency_metrics(study, best_k_trials):
    """
    Calculates efficiency metrics for the top K trials from a study.

    Args:
        study: The Optuna study object containing all trial information.
        best_k_trials: A DataFrame or similar object containing the best K trials.

    Returns:
        A dictionary containing the calculated efficiency metrics for each trial.
    """
    # Get the indices of the best k trials
    idx_trials = best_k_trials.index.tolist()

    # Initialize a dictionary to store the results
    results = {}

    # Iterate over the indices of the trials
    for i in idx_trials:
        # Get the model for the corresponding trial
        trial = study.get_trials(deepcopy=True)[i]
        # Get the parameters of the model
        params_model = trial.params

        # Extract the corresponding parameters from the trial
        n_layers = params_model["n_layers"]
        # Extract the number of filters for each layer
        n_filters = [params_model[f"n_filters_layer{i}"] for i in range(n_layers)]
        # Extract the kernel sizes for each layer
        kernel_sizes = [params_model[f"kernel_size_layer{i}"] for i in range(n_layers)]
        # Extract the dropout rate
        dropout_rate = params_model["dropout_rate"]
        # Extract the size of the fully connected layer
        fc_size = params_model["fc_size"]

        # Create a new model instance with the extracted parameters
        model_trial =  FlexibleCNN(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate,
            fc_size=fc_size,
            num_classes=2
        )

        # Initialize the classifier to count its parameters
        resolution = params_model["resolution"]
        dummy_input = torch.randn(1, 3, resolution, resolution)
        model_trial(dummy_input)

        # Get the efficiency metrics for the model
        total_trainable_params = get_trainable_params(model_trial)

        # Get the accuracy of the model from the trial's value
        accuracy = trial.value

        # Store the trial's metrics in the results dictionary
        results[i] = {
            'trial': i,
            "model_size": total_trainable_params,
            "accuracy": accuracy,
        }

    # Return the dictionary of results
    return results

#%% 
def plot_model_metrics(results_df):
    """
    Args:
        results_df: A pandas DataFrame containing the results, with columns for 'accuracy',
                    'model_size', and a unique identifier for each trial.
    """
    # Define the column to be used for labeling points
    label_column = 'trial'
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the unique identifiers for each data point
    unique_labels = results_df[label_column].unique()
    
    # Get a color palette from matplotlib
    base_colors = plt.get_cmap("tab10").colors
    # Create a list of colors, cycling through the palette if needed
    colors = [base_colors[i % len(base_colors)] for i in range(len(unique_labels))]
    # Map each unique label to a specific color
    label_color_map = dict(zip(unique_labels, colors))

    # Iterate through the DataFrame to plot each data point
    for _, row in results_df.iterrows():
        # Plot a single point with its accuracy and model size
        ax.scatter(
            row["accuracy"],
            row["model_size"],
            color=label_color_map[row[label_column]],
            label=row[label_column]
        )

    # Set the title of the plot
    ax.set_title("Accuracy vs Model Size")
    # Set the label for the y-axis
    ax.set_ylabel("Model Size")
    # Set the label for the x-axis
    ax.set_xlabel("Accuracy")

    # Add a legend to the plot with a single entry for each label
    handles = []
    # Create a set to track labels that have been added to the legend
    added_labels = set()
    # Iterate through the unique labels and their colors to create legend handles
    for label, color in label_color_map.items():
        # Create a proxy artist for the legend entry
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, label=label))
    # Display the legend on the axes
    ax.legend(handles=handles, title=label_column, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust plot parameters for a tight layout
    plt.tight_layout()
    # Display the plot
    plt.show()
    
#%% 
results = add_efficiency_metrics(study, best_k_trials)

results_df = pd.DataFrame(results).T
results_df = results_df.astype({'trial': 'int64', 'model_size': 'int64'})

plot_model_metrics(results_df)

#%%  Evaluating the model with alternative metrics
def train_with_alternative_metrics(study, best_k_trials, device, dataset_path):
    """
    Trains a selection of models and evaluates them using multiple metrics.

    Args:
        study: The Optuna study object.
        best_k_trials: A DataFrame of the best K trials to be re-evaluated.
        device: The device (e.g., 'cpu' or 'cuda') to perform training and evaluation on.
        dataset_path: The file path to the dataset.

    Returns:
        A dictionary containing the evaluation metrics for each trial.
    """
    # Extract the indices of the best k trials
    idx_trials = best_k_trials.index.tolist()
    # Initialize a dictionary to store the results
    results = {}

    # Iterate over the indices of the best trials
    for i in idx_trials:
        # Get the trial object from the study
        trial = study.get_trials(deepcopy=True)[i]
        # Extract the model parameters from the trial
        params_model = trial.params

        # Extract model parameters from the trial's parameters
        n_layers = params_model["n_layers"]
        # Create a list of the number of filters for each convolutional layer
        n_filters = [params_model[f"n_filters_layer{j}"] for j in range(n_layers)]
        # Create a list of the kernel sizes for each convolutional layer
        kernel_sizes = [params_model[f"kernel_size_layer{j}"] for j in range(n_layers)]
        # Extract the dropout rate
        dropout_rate = params_model["dropout_rate"]
        # Extract the size of the fully connected layer
        fc_size = params_model["fc_size"]

        # Instantiate a new model with the extracted parameters
        model_trial = FlexibleCNN(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate,
            fc_size=fc_size,
            num_classes=2
        )

        # Initialize dynamic layers before defining the optimizer
        resolution = params_model["resolution"]
        dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
        model_trial = model_trial.to(device)
        model_trial(dummy_input)

        # Initialize the optimizer for the model
        optimizer = optim.Adam(model_trial.parameters(), lr=params_model['learning_rate'])
        # Define the loss function
        loss_fcn = nn.CrossEntropyLoss()

        # Get the data loaders for training and validation
        train_loader, val_loader = get_data_loaders(
            transforms.Compose([
                transforms.Resize((params_model["resolution"], params_model["resolution"])),
                transforms.ToTensor(),
            ]),
            params_model["batch_size"],
            AIvsReal_path=dataset_path
        )

        # Set the number of training epochs
        n_epochs = 3
        # Move the model to the specified device
        model_trial = model_trial.to(device)

        # Begin the training loop
        for epoch in range(n_epochs):
            # Run a single training epoch
            _ = training_epoch(
                model_trial,
                train_loader,
                optimizer,
                loss_fcn,
                device,
                epoch,
                n_epochs,
                silent=False,
            )

        # Initialize the evaluation metrics
        accuracy_metric = Accuracy(task="binary").to(device)
        precision_metric = Precision(task="binary").to(device)
        recall_metric = Recall(task="binary").to(device)

        # Set the model to evaluation mode
        model_trial.eval()
        # Disable gradient calculations for evaluation
        with torch.no_grad():
            # Iterate through the validation data
            for inputs, labels in val_loader:
                # Move data to the specified device
                inputs, labels = inputs.to(device), labels.to(device)
                # Perform a forward pass
                outputs = model_trial(inputs)
                # Get the predicted class
                preds = torch.argmax(outputs, dim=1)

                # Update the metrics with the current batch's predictions and labels
                accuracy_metric.update(preds, labels)
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)

        # Compute the final metric scores and store them in the results dictionary
        results[i] = {
            'trial': i,
            'accuracy': accuracy_metric.compute().item(),
            'precision': precision_metric.compute().item(),
            'recall': recall_metric.compute().item(),
        }

    # Return the dictionary of results
    return results

#%%
# NOTE: It takes about 16 minutes to run
results_alt = train_with_alternative_metrics(study=study, best_k_trials=best_k_trials, device=DEVICE, dataset_path=AIvsReal_path)

#%%
def plot_metric_scatter(df, x_col="accuracy", y_col="precision", color_col="recall", label_col="trial"):
    """
    Creates a scatter plot to visualize the relationship between two metrics,
    with a third metric represented by color.

    Args:
        df: A pandas DataFrame containing the data to plot.
        x_col: The name of the column to use for the x-axis.
        y_col: The name of the column to use for the y-axis.
        color_col: The name of the column to use for point colors.
        label_col: The name of the column to use for labeling each point.
    """
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize color values and define the colormap
    color_values = df[color_col]
    # Create a normalization object for the color range
    norm = mpl.colors.Normalize(vmin=color_values.min(), vmax=color_values.max())
    # Define the colormap to be used
    cmap = plt.cm.viridis

    # Create the scatter plot
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=100,
        edgecolor='k',
        alpha=0.8
    )

    # Add a colorbar to the plot
    cbar = plt.colorbar(scatter, ax=ax)
    # Set the label for the colorbar
    cbar.set_label(color_col.capitalize())

    # Add text labels to each point on the plot
    for _, row in df.iterrows():
        ax.text(
            row[x_col],
            row[y_col],
            str(row[label_col]),
            fontsize=9,
            ha='center',
            va='center',
            color='white',
        )

    # Set the labels and title of the plot
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.set_title(f"{y_col.capitalize()} vs {x_col.capitalize()} (colored by {color_col.capitalize()})")
    # Adjust plot parameters for a tight layout
    plt.tight_layout()
    # Display the plot
    plt.show()
    
#%% 
results_df_metrics = pd.DataFrame(results_alt).T
results_df_metrics = results_df_metrics.astype({'trial': 'Int64'})

print(results_df_metrics)

# get the best model based on precision
best_precision_trial = results_df_metrics.loc[results_df_metrics['precision'].idxmax()]
print(f"Best model based on precision: Trial {best_precision_trial['trial']} with Precision: {best_precision_trial['precision']:.4f}, Recall: {best_precision_trial['recall']:.4f}, Accuracy: {best_precision_trial['accuracy']:.4f}") 