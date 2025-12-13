#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:55:58 2025

@author: ek
"""

import torch
import numpy as np
import pandas as pd

#%% 
# From Python lists
x = torch.tensor([1, 2, 3])

print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)

# From a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)

print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)

# From Pandas DataFrame
# Read the data from the CSV file into a DataFrame
df = pd.read_csv('./data.csv')

# Extract the data as a NumPy array from the DataFrame
all_values = df.values

# Convert the DataFrame's values to a PyTorch tensor
tensor_from_df = torch.tensor(all_values)

print("ORIGINAL DATAFRAME:\n\n", df)
print("\nRESULTING TENSOR:\n\n", tensor_from_df)
print("\nTENSOR DATA TYPE:", tensor_from_df.dtype)

# All zeros
zeros = torch.zeros(2, 3)
print("TENSOR WITH ZEROS:\n\n", zeros)

# All ones
ones = torch.ones(2, 3)
print("TENSOR WITH ONES:\n\n", ones)

# Random numbers
random = torch.rand(2, 3)
print("RANDOM TENSOR:\n\n", random)

# Range of numbers
range_tensor = torch.arange(0, 10, step=1)
print("ARANGE TENSOR:", range_tensor)

# A 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)

#%% Manipulations
print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Add dimension
expanded = x.unsqueeze(0)  # Add dimension at index 0

print("\nTENSOR WITH ADDED DIMENSION AT INDEX 0:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)

print("EXPANDED TENSOR:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)
print("-"*45)

# Remove dimension
squeezed = expanded.squeeze()

print("\nTENSOR WITH DIMENSION REMOVED:\n\n", squeezed)
print("\nTENSOR SHAPE:", squeezed.shape)

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Reshape
reshaped = x.reshape(3, 2)

print("\nAFTER PERFORMING reshape(3, 2):\n\n", reshaped)
print("\nTENSOR SHAPE:", reshaped.shape)

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Transpose
transposed = x.transpose(0, 1)

print("\nAFTER PERFORMING transpose(0, 1):\n\n", transposed)
print("\nTENSOR SHAPE:", transposed.shape)

#%% Concatenate
# Create two tensors to concatenate
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])

# Concatenate along columns (dim=1)
concatenated_tensors = torch.cat((tensor_a, tensor_b), dim=1)


print("TENSOR A:\n\n", tensor_a)
print("\nTENSOR B:\n\n", tensor_b)
print("-"*45)
print("\nCONCATENATED TENSOR (dim=1):\n\n", concatenated_tensors)

#%% Accessing elements
# Create a 3x4 tensor
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get a single element at row 1, column 2
single_element_tensor = x[1, 2]

print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)
print("-" * 55)

# Get the entire second row (index 1)
second_row = x[1]

print("\nINDEXING ENTIRE ROW [1]:", second_row)
print("-" * 55)

# Last row
last_row = x[-1]

print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row, "\n")

#%% Slicing
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get the first two rows
first_two_rows = x[0:2]

print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)
print("-" * 55)

# Get the third column of all rows
third_column = x[:, 2]

print("\nSLICING THIRD COLUMN ([:, 2]]):", third_column)
print("-" * 55)

# Every other column
every_other_col = x[:, ::2]

print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)
print("-" * 55)

# Last column
last_col = x[:, -1]

print("\nLAST COLUMN ([:, -1]):", last_col, "\n")

#%% Combining indexing and slicing
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Combining slicing and indexing (First two rows, last two columns)
combined = x[0:2, 2:]

print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined, "\n")

#%% .item(): Extracts the value from a single-element tensor as a standard Python number.
print("SINGLE-ELEMENT TENSOR:", single_element_tensor)
print("-" * 45)

# Extract the value from a single-element tensor as a standard Python number
value = single_element_tensor.item()

print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))

#%% Boolean Masking: Using a boolean tensor to select elements that meet a certain condition (e.g., x[x > 5]).
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Boolean indexing using logical comparisons
mask = x > 6

print("MASK (VALUES > 6):\n\n", mask, "\n")

# Applying Boolean masking
mask_applied = x[mask]

print("VALUES AFTER APPLYING MASK:", mask_applied, "\n")

#%% Fancy Indexing: Using a tensor of indices to select specific elements in a non-contiguous way.
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Fancy indexing

# Get first and third rows
row_indices = torch.tensor([0, 2])

# Get second and fourth columns
col_indices = torch.tensor([1, 3]) 

# Gets values at (0,1), (0,3), (2,1), (2,3)

get_values = x[row_indices[:, None], col_indices]

print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values, "\n")

#%% Element-wise Operations: Standard math operators (+, *) that apply to each element independently.
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 60)

# Element-wise addition
element_add = a + b

print("\nAFTER PERFORMING ELEMENT-WISE ADDITION:", element_add, "\n")

#%% # Element-wise multiplication
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

element_mul = a * b
print("\nAFTER PERFORMING ELEMENT-WISE MULTIPLICATION:", element_mul, "\n")

#%% Dot Product (torch.matmul()): Calculates the dot product of two vectors or matrices.
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

# Dot product
dot_product = torch.matmul(a, b)

print("\nAFTER PERFORMING DOT PRODUCT:", dot_product, "\n")

#%% Broadcasting: The automatic expansion of smaller tensors to match the shape of larger tensors during arithmetic operations.
# Broadcasting allows operations between tensors with compatible shapes, even if they don't have the exact same dimensions.
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1],
                 [2],
                 [3]])

print("TENSOR A:", a)
print("SHAPE:", a.shape)
print("\nTENSOR B\n\n", b)
print("\nSHAPE:", b.shape)
print("-" * 65)

# Apply broadcasting
c = a + b


print("\nTENSOR C:\n\n", c)
print("\nSHAPE:", c.shape, "\n")

#%% Comparison Operators: Element-wise comparisons (>, ==, <) that produce a boolean tensor.
temperatures = torch.tensor([20, 35, 19, 35, 42])
print("TEMPERATURES:", temperatures)
print("-" * 50)

### Comparison Operators (>, <, ==)

# Use '>' (greater than) to find temperatures above 30
is_hot = temperatures > 30

# Use '<=' (less than or equal to) to find temperatures 20 or below
is_cool = temperatures <= 20

# Use '==' (equal to) to find temperatures exactly equal to 35
is_35_degrees = temperatures == 35

print("\nHOT (> 30 DEGREES):", is_hot)
print("COOL (<= 20 DEGREES):", is_cool)
print("EXACTLY 35 DEGREES:", is_35_degrees, "\n")

#%% Logical Operators: Element-wise logical operations (& for AND, | for OR) on boolean tensors.
is_morning = torch.tensor([True, False, False, True])
is_raining = torch.tensor([False, False, True, True])
print("IS MORNING:", is_morning)
print("IS RAINING:", is_raining)
print("-" * 50)

### Logical Operators (&, |)

# Use '&' (AND) to find when it's both morning and raining
morning_and_raining = (is_morning & is_raining)

# Use '|' (OR) to find when it's either morning or raining
morning_or_raining = is_morning | is_raining

print("\nMORNING & (AND) RAINING:", morning_and_raining)
print("MORNING | (OR) RAINING:", morning_or_raining)

#%% torch.mean(): Calculates the mean of all elements in a tensor.
data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
print("DATA:", data)
print("-" * 45)

# Calculate the mean
data_mean = data.mean()

print("\nCALCULATED MEAN:", data_mean, "\n")

#%% torch.std(): Calculates the standard deviation of all elements.
print("DATA:", data)
print("-" * 45)

# Calculate the standard deviation
data_std = data.std()

print("\nCALCULATED STD:", data_std, "\n")

#%% Type Casting (.int, etc.): Converts a tensor from one data type to another (e.g., from float to integer).
print("DATA:", data)
print("DATA TYPE:", data.dtype)
print("-" * 45)

# Cast the tensor to a int type
int_tensor = data.int()

print("\nCASTED DATA:", int_tensor)
print("CASTED DATA TYPE", int_tensor.dtype)

#%% Exercise 1: Analyzing Monthly Sales Data
# You're a data analyst at an e-commerce company. You've been given a tensor representing the monthly sales of three different products over a period of four months. Your task is to extract meaningful insights from this data.

# The tensor sales_data is structured as follows:

# Rows represent the products (Product A, Product B, Product C).

# Columns represent the months (Jan, Feb, Mar, Apr).

# Your goals are:

# Calculate the total sales for Product B (the second row).
# Identify which months had sales greater than 130 for Product C (the third row) using boolean masking.
# Extract the sales data for all products for the months of Feb and Mar (the middle two columns).

# Sales data for 3 products over 4 months
sales_data = torch.tensor([[100, 120, 130, 110],   # Product A
                           [ 90,  95, 105, 125],   # Product B
                           [140, 115, 120, 150]    # Product C
                          ], dtype=torch.float32)

print("ORIGINAL SALES DATA:\n\n", sales_data)
print("-" * 45)

### START CODE HERE ###

# 1. Calculate total sales for Product B.
total_sales_product_b = torch.sum(sales_data[1, :])
total_sales_product_b = sales_data[1].sum()

# 2. Find months where sales for Product C were > 130.
high_sales_mask_product_c = sales_data[2] > 130

# 3. Get sales for Feb and Mar for all products.
sales_feb_mar = sales_data[:, 1:3]

### END CODE HERE ###

print("\nTotal Sales for Product B:                   ", total_sales_product_b)
print("\nMonths with >130 Sales for Product C (Mask): ", high_sales_mask_product_c)
print("\nSales for Feb & Mar:\n\n", sales_feb_mar)

#%% Exercise 2: Image Batch Transformation
# You're working on a computer vision model and have a batch of 4 grayscale images, each of size 3x3 pixels. The data is currently in a tensor with the shape [4, 3, 3], which represents [batch_size, height, width].

# For processing with certain deep learning frameworks, you need to transform this data into the [batch_size, channels, height, width] format. Since the images are grayscale, you'll need to:

# Add a new dimension of size 1 at index 1 to represent the color channel.
# After adding the channel, you realize the model expects the shape [batch_size, height, width, channels]. Transpose the tensor to swap the channel dimension with the last dimension.

# A batch of 4 grayscale images, each 3x3
image_batch = torch.rand(4, 3, 3)

print("ORIGINAL BATCH SHAPE:", image_batch.shape)
print("-" * 45)

### START CODE HERE ###

# 1. Add a channel dimension at index 1.
image_batch_with_channel = torch.reshape(image_batch, (4, 1, 3, 3))
# Alt: image_batch_with_channel = image_batch.unsqueeze(1)

# 2. Transpose the tensor to move the channel dimension to the end.
# Swap dimension 1 (channels) with dimension 3 (the last one).
image_batch_transposed = torch.transpose(image_batch_with_channel, 1, 3)
# Alt: image_batch_transposed = torch.transpose(image_batch_with_channel, 1, 3)

### END CODE HERE ###


print("\nSHAPE AFTER UNSQUEEZE:", image_batch_with_channel.shape)
print("SHAPE AFTER TRANSPOSE:", image_batch_transposed.shape)

#%% Exercise 3: Combining and Weighting Sensor Data
# You're building an environment monitoring system that uses two sensors: one for temperature and one for humidity. You receive data from these sensors as two separate 1D tensors.

# Your task is to:

# Concatenate the two tensors into a single 2x5 tensor, where the first row is temperature data and the second is humidity data.
# Create a weights tensor torch.tensor([0.6, 0.4]).
# Use broadcasting and element-wise multiplication to apply these weights to the combined sensor data. The temperature data should be multiplied by 0.6 and the humidity data by 0.4.
# Finally, calculate the weighted average for each time step by summing the weighted values along dim=0 and dividing by the sum of the weights.

# Sensor readings (5 time steps)
temperature = torch.tensor([22.5, 23.1, 21.9, 22.8, 23.5])
humidity = torch.tensor([55.2, 56.4, 54.8, 57.1, 56.8])

print("TEMPERATURE DATA: ", temperature)
print("HUMIDITY DATA:    ", humidity)
print("-" * 45)

### START CODE HERE ###

# 1. Concatenate the two tensors.
# Note: You need to unsqueeze them first to stack them vertically.
combined_data = torch.cat((temperature.unsqueeze(0), humidity.unsqueeze(0)))

# 2. Create the weights tensor.
weights = torch.tensor([0.6, 0.4])

# 3. Apply weights using broadcasting.
# You need to reshape weights to [2, 1] to broadcast across columns.
weighted_data = combined_data * weights.reshape((2, 1))

# 4. Calculate the weighted average for each time step.
#    (A true average = weighted sum / sum of weights)
weighted_sum = weighted_data.sum(dim=0)
weighted_average = weighted_sum / weights.sum()

### END CODE HERE ###

print("\nCOMBINED DATA (2x5):\n\n", combined_data)
print("\nWEIGHTED DATA:\n\n", weighted_data)
print("\nWEIGHTED AVERAGE:", weighted_average)

#%% Exercise 4: Feature Engineering for Taxi Fares
# You are working with a dataset of taxi trips. You have a tensor, trip_data, where each row is a trip and the columns represent [distance (km), hour_of_day (24h)].

# Your goal is to engineer a new binary feature called is_rush_hour_long_trip. This feature should be True (or 1) only if a trip meets both of the following criteria:

# It's a long trip (distance > 10 km).
# It occurs during a rush hour (8-10 AM or 5-7 PM, i.e., [8, 10) or [17, 19)).
# To achieve this, you will need to:

# Slice the trip_data tensor to isolate the distance and hour columns.
# Use logical and comparison operators to create boolean masks for each condition (long trip, morning rush, evening rush).
# Combine these masks to create the final is_rush_hour_long_trip feature.
# Reshape this new 1D feature tensor into a 2D column vector and convert its data type to float so it can be combined with the original data.

# Data for 8 taxi trips: [distance, hour_of_day]
trip_data = torch.tensor([
    [5.3, 7],   # Not rush hour, not long
    [12.1, 9],  # Morning rush, long trip -> RUSH HOUR LONG
    [15.5, 13], # Not rush hour, long trip
    [6.7, 18],  # Evening rush, not long
    [2.4, 20],  # Not rush hour, not long
    [11.8, 17], # Evening rush, long trip -> RUSH HOUR LONG
    [9.0, 9],   # Morning rush, not long
    [14.2, 8]   # Morning rush, long trip -> RUSH HOUR LONG
], dtype=torch.float32)


print("ORIGINAL TRIP DATA (Distance, Hour):\n\n", trip_data)
print("-" * 55)


### START CODE HERE ###

# 1. Slice the main tensor to get 1D tensors for each feature.
distances = trip_data[:, 0]
hours = trip_data[:, 1]

# 2. Create boolean masks for each condition.
is_long_trip = distances > 10
is_morning_rush = (hours >= 8) & (hours < 10)
is_evening_rush = (hours >= 17) & (hours < 19)

# 3. Combine masks to identify rush hour long trips.
# A trip is a rush hour long trip if it's (a morning OR evening rush) AND a long trip.
is_rush_hour_long_trip_mask = (is_morning_rush | is_evening_rush) & is_long_trip

# 4. Reshape the new feature into a column vector and cast to float.
new_feature_col = is_rush_hour_long_trip_mask.float().unsqueeze(1)

### END CODE HERE ###

print("\n'IS RUSH HOUR LONG TRIP' MASK: ", is_rush_hour_long_trip_mask)
print("\nNEW FEATURE COLUMN (Reshaped):\n\n", new_feature_col)

# You can now concatenate this new feature to the original data
enhanced_trip_data = torch.cat((trip_data, new_feature_col), dim=1)
print("\nENHANCED DATA (with new feature at the end):\n\n", enhanced_trip_data)