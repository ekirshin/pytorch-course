#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:31:45 2026

@author: ek
"""
import gc
import os
import json

import torch
from torch.utils.data import DataLoader

import helper_utils

#%% 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% Load the CIFAR10 dataset.
trainset = helper_utils.download_and_load_cifar10()

#%% find out exactly how many cores your environment has by running
cpu_cores = os.cpu_count()
print(f"Number of available CPU cores: {cpu_cores}")

#%%
# Define the list of num_workers values to test
# workers_to_test = [0, 2, 4, 6, 8, 10]
workers_to_test = list(range(0, 33, 2))

#%% 
def experiment_workers(workers_to_test, trainset, device):
    """
    Measures the data loading time for different numbers of workers.

    Args:
        workers_to_test: A list of integers representing the number of workers to test.
        trainset: The dataset to be loaded.
        device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
    """
    # Initialize a dictionary to store the results
    worker_times = {}

    # Loop through each worker number you want to test
    for nw in workers_to_test:
        print(f"--- Testing Number of Workers = {nw} ---")
        
        # Create a new DataLoader instance for each specific test.
        loader = DataLoader(trainset, 
                            batch_size=32, 
                            shuffle=True,
                            # The 'num_workers' is set to the current value in the loop.
                            num_workers=nw
                        )
        
        # Handle potential runtime errors
        try:
            # Time the data loading for one epoch and save it to the dictionary
            worker_times[nw] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # If an error occurs (often from running out of shared memory)
            print(f"\n❌ ERROR with {nw} workers. Likely a shared memory issue.")
            worker_times[nw] = float('inf')
            
        # Clean up the loader and call the garbage collector to free up memory
        del loader
        gc.collect()

        # Clear the PyTorch CUDA cache to free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return worker_times

#%% 
# Run the experiment to measure the data loading time for different numbers of workers.
worker_times = helper_utils.run_experiment(
    # A unique name for this experiment, used as the filename for the cached results.
    experiment_name='worker_times', 
    # The actual function that contains the experiment's logic.
    experiment_fcn=experiment_workers, 
    # The parameters to iterate over; in this case, a list of worker counts.
    cases=workers_to_test, 
    # The dataset required by the experiment function.
    trainset=trainset, 
    # The computation device (e.g., 'cpu' or 'cuda') to be used.
    device=device,
    # If False, the function will load results from the cache if they exist.
    # If True, it will force the experiment to run again and overwrite any old results.
    rerun=False
)

#%% 
helper_utils.plot_performance_summary(
    worker_times, 
    title="DataLoader Performance vs. num_workers", 
    xlabel="Number of Workers", 
    ylabel="Average Time per Epoch (milliseconds)"
)

#%%
### This cell will take a few seconds to run

# Create the dictionary of loaders iteratively using a dictionary comprehension
# for each number in the 'workers_to_test' list.
loaders_to_compare = {
    f"{nw} Workers": DataLoader(trainset, batch_size=32, num_workers=nw) 
    for nw in workers_to_test
}

# Pass the generated dictionary to the plotting function.
helper_utils.visualize_dataloader_efficiency(loaders_to_compare, device)

# Clean up and release memory.
del loaders_to_compare
gc.collect()

# Clear the PyTorch CUDA cache to free up GPU memory.
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
#%% Exploring the Effect of batch_size
# Define the list of batch_size values to test
batch_sizes_to_test = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

#%% 
def experiment_batch_sizes(batch_sizes_to_test, trainset, device):
    """
    Measures the data loading time for different batch sizes.

    Args:
        batch_sizes_to_test: A list of integers representing the batch sizes to test.
        trainset: The dataset to be loaded.
        device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
    """
    # Initialize a dictionary to store the results
    batch_size_times = {}

    # Loop through each batch size you want to test
    for bs in batch_sizes_to_test:
        print(f"--- Testing Batch Size = {bs} ---")
        
        # Create a new DataLoader instance for each specific test.
        loader = DataLoader(trainset, 
                            # The 'batch_size' is set to the current value in the loop.
                            batch_size=bs, 
                            shuffle=True,
                            num_workers=6
                        )
        
        # Handle potential runtime errors, especially out-of-memory
        try:
            # Time the data loading for one epoch and save it to the dictionary
            batch_size_times[bs] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # If an error occurs (often from running out of GPU memory),
            print(f"\n❌ ERROR with batch size {bs}. Likely a GPU memory issue.")
            batch_size_times[bs] = float('inf')
            
        # Clean up the loader and call the garbage collector to free up memory
        # ensuring each test runs in a clean environment.
        del loader
        gc.collect()

        # Clear the PyTorch CUDA cache to free up GPU memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return batch_size_times

#%% # Run the experiment to measure the data loading time for different batch sizes.
batch_size_times = helper_utils.run_experiment(
    # A unique name for this experiment, used as the filename for the cached results.
    experiment_name="batch_size_times", 
    # The actual function that contains the experiment's logic.
    experiment_fcn=experiment_batch_sizes,
    # The parameters to iterate over; in this case, a list of different batch sizes.
    cases=batch_sizes_to_test,
    # The dataset required by the experiment function.
    trainset=trainset,
    # The computation device (e.g., 'cpu' or 'cuda') to be used.
    device=device,
    # If False, the function will load results from the cache if they exist.
    # If True, it will force the experiment to run again and overwrite any old results.
    rerun=False
)

#%% Visualize thre results
helper_utils.plot_performance_summary(
    batch_size_times, 
    title="DataLoader Performance vs. batch_size", 
    xlabel="Batch Sizes", 
    ylabel="Average Time per Epoch (milliseconds)"
)

#%% Optimizing with pin_memory
pin_memory_settings = [False, True]

#%% 
def experiment_pin_memory(pin_memory_settings, trainset, device):
    """
    Measures the data loading time with and without pinned memory.

    Args:
        pin_memory_settings: A list of boolean values to test for pin_memory.
        trainset: The dataset to be loaded.
        device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
    """
    # Initialize a dictionary to store the results
    pin_memory_times = {}

    # Loop through each pin_memory setting
    for setting in pin_memory_settings:
        print(f"--- Testing with pin_memory = {setting} ---")
        
        # Create a DataLoader with the current pin_memory setting
        loader = DataLoader(trainset,
                            batch_size=256,
                            num_workers=10,
                            shuffle=True,
                            # The 'pin_memory' is set to the current boolean value in the loop.
                            pin_memory=setting
                        )
        
        try:
            # Measure performance and store the result in the dictionary
            pin_memory_times[setting] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # Print an error message if an exception occurs
            print(f"\n❌ An error occurred with pin_memory = {setting}: {e}")
            pin_memory_times[setting] = float('inf')
            
        # --- Memory Cleanup for each iteration ---
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pin_memory_times

#%% # Run the experiment to measure the data loading time for pin memory either set as False or True.
pin_memory_times = helper_utils.run_experiment(
    # A unique name for this experiment, used as the filename for the cached results.
    experiment_name="pin_memory_times",
    # The actual function that contains the experiment's logic.
    experiment_fcn=experiment_pin_memory,
    # The parameters to iterate over; in this case, a list of boolean values for pin memory.
    cases=pin_memory_settings,
    # The dataset required by the experiment function.
    trainset=trainset, 
    # The computation device (e.g., 'cpu' or 'cuda') to be used.
    device=device,
    # If False, the function will load results from the cache if they exist.
    # If True, it will force the experiment to run again and overwrite any old results.
    rerun=False
)

#%% 
helper_utils.plot_performance_summary(
    pin_memory_times, 
    title="DataLoader Performance vs. pin_memory", 
    xlabel="Pin Memory", 
    ylabel="Average Time per Epoch (milliseconds)"
)

#%% Fine-Tuning with prefetch_factor
# Define the list of prefetch_factor values to test
prefetch_factors_to_test = [2, 4, 6, 8, 10, 12]

#%% 
def experiment_prefetch_factor(prefetch_factors_to_test, trainset, device):
    """
    Measures the data loading time for different prefetch factor settings.

    Args:
        prefetch_factors_to_test: A list of integers representing the prefetch factors to test.
        trainset: The dataset to be loaded.
        device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
    """
    # Initialize a dictionary to store the results
    prefetch_factor_times = {}

    # Loop through each prefetch factor you want to test
    for pf in prefetch_factors_to_test:
        print(f"--- Testing prefetch_factor = {pf} ---")
        
        # Create a new DataLoader instance for each specific test, using the optimal settings
        loader = DataLoader(trainset, 
                            batch_size=256, 
                            shuffle=True,
                            num_workers=6,
                            pin_memory=False,
                            # The 'prefetch_factor' is set to the current value in the loop.
                            prefetch_factor=pf
                        )
        
        # Handle potential runtime errors
        try:
            # Time the data loading for one epoch and save it to the dictionary
            prefetch_factor_times[pf] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # If an error occurs, record it.
            print(f"\n❌ ERROR with prefetch_factor {pf}: {e}")
            prefetch_factor_times[pf] = float('inf')
            
        # Clean up the loader and call the garbage collector to free up memory
        # ensuring each test runs in a clean environment.
        del loader
        gc.collect()

        # Clear the PyTorch CUDA cache to free up GPU memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return prefetch_factor_times

#%% # Run the experiment to measure the data loading time for different prefetch factor.
prefetch_factor_times = helper_utils.run_experiment(
    # A unique name for this experiment, used as the filename for the cached results.
    experiment_name="prefetch_factor_times", 
    # The actual function that contains the experiment's logic.
    experiment_fcn=experiment_prefetch_factor,
    # The parameters to iterate over; in this case, a list of different prefetch factor.
    cases=prefetch_factors_to_test,
    # The dataset required by the experiment function.
    trainset=trainset, 
    # The computation device (e.g., 'cpu' or 'cuda') to be used.
    device=device,
    # If False, the function will load results from the cache if they exist.
    # If True, it will force the experiment to run again and overwrite any old results.
    rerun=False
)

#%% 
helper_utils.plot_performance_summary(
    prefetch_factor_times, 
    title="DataLoader Performance vs. prefetch_factor", 
    xlabel="Prefetch Factor", 
    ylabel="Average Time per Epoch (milliseconds)"
)

#%% Custom experiment
trainset = helper_utils.download_and_load_cifar10()

#%% 
def custom_experiment(trainset, device):
    """
    Runs a custom experiment to measure DataLoader performance.

    Args:
        trainset: The dataset to be used for the experiment.
        device: The device (e.g., 'cpu' or 'cuda') on which to run the test.

    Returns:
        A tuple containing:
            - A dictionary with the performance results.
            - The name of the parameter that was tested.
    """
    
    # Specify the name of the DataLoader parameter to be tested.
    # For example: parameter_name = 'prefetch_factor'
    
    parameter_name = 'prefetch_factor'

    # Provide a list of values to iterate through for the specified parameter.
    # For example: list_of_values_to_test = [6, 8]
    
    list_of_values_to_test = [2, 4, 6, 8]

    # Initialize an empty dictionary to store the performance results.
    results_dictionary = {}

    # Iterate over each value in the test list.
    for current_value in list_of_values_to_test:
        print(f"--- Testing {parameter_name} = {current_value} ---")
        
        # Configure and instantiate the DataLoader for the current test iteration.
        # For example: loader = DataLoader(trainset, 
                                       #  batch_size=64, 
                                       #  shuffle=True, 
                                       #  num_workers=2, 
                                       #  pin_memory=False,
                                       #  prefetch_factor=current_value
                                       # )
        
        loader = DataLoader(trainset, 
                            batch_size=64, 
                            shuffle=True, 
                            num_workers=2, 
                            pin_memory=True,
                            prefetch_factor=current_value
                           )
        
        # Measure the performance and handle potential runtime errors.
        try:
            # Calculate the average epoch time and store it in the results dictionary.
            results_dictionary[current_value] = helper_utils.measure_average_epoch_time(loader, device)
        except RuntimeError as e:
            # Handle cases where a runtime error occurs, such as an out-of-memory issue.
            print(f"\n❌ ERROR with {parameter_name} = {current_value}: {e}")
            results_dictionary[current_value] = float('inf')
            
        # Ensure each test run is independent by cleaning up memory.
        # Clean up the DataLoader instance to free up resources.
        del loader
        # Invoke the garbage collector to release unreferenced memory.
        gc.collect()

        # Clear the CUDA cache if a GPU is available.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return the dictionary of results and the name of the tested parameter.
    return results_dictionary, parameter_name

#%% 
results_dictionary, parameter_name = custom_experiment(trainset=trainset, device=device)

helper_utils.plot_performance_summary(
    results_dictionary, 
    title=f"DataLoader Performance vs. {parameter_name}", 
    xlabel=parameter_name.replace('_', ' ').title(), 
    ylabel="Average Time per Epoch (milliseconds)"
)