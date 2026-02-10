#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab 4.
Note: model file is in the Work/Training storage.

Created on Sun Feb  1 18:04:58 2026

@author: ek
"""
import random

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers

import helper_utils

import gzip
import shutil
import os

# Set random seed for reproducibility
SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#%% 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%%

def decompress_gzip(input_path, output_path):
    # Only decompress if the output file does not already exist
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping decompression.")
        return

    print(f"Decompressing {input_path} → {output_path}")
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

#%% Prep dataset
decompress_gzip("recipes_fruit_veg.csv.gz", "recipes_fruit_veg.csv")

#%% # Load the filtered dataset into a pandas DataFrame
df = pd.read_csv("recipes_fruit_veg.csv")

# Create the numerical 'label' column: 0 for 'fruit', 1 for 'vegetable'
df['label'] = 1
df.loc[df['category'] == 'fruit', 'label'] = 0

# Extract the recipe names and labels into lists
df_clean = df.dropna(subset=['name'])
texts = df_clean['name'].tolist()
labels = df_clean['label'].tolist()

# Verify the dataset size and class distribution
print(f"Total samples for classification:  {len(texts)}")
print(f"Fruit recipes:                     {labels.count(0)}, {round(labels.count(0)/(labels.count(0) + labels.count(1)) *100,1)} %")
print(f"Vegetable recipes:                 {labels.count(1)}, {round(labels.count(1)/(labels.count(0) + labels.count(1)) *100,1)} %")

#%% Preview data
# Set the number of random samples to display.
num_samples = 10

# Display a sample of name and label pairs.
display(df[['name', 'label']].sample(num_samples, random_state=25).style.hide(axis="index"))

sample_df = df[['name', 'label']].sample(num_samples, random_state=25)

#%% Loading the Pre-trained Transformer
model_name="distilbert-base-uncased"
model_path="./distilbert-local-base"

# Ensure the model is downloaded
helper_utils.download_bert(model_name, model_path)    

# Load the pre-trained transformer: num_classes=2: Attaches a new randomly initialized classification head with 2 output labels, preparing the model for your binary classification task.
bert_model, bert_tokenizer = helper_utils.load_bert(model_path, num_classes=2)

#%% Preparing Data for Training
# Now that you have your model, tokenizer, and data lists ready, the next step is to structure this data into the objects PyTorch requires for training. This process is simpler than in the previous lab because many of the manual steps you performed before, such as cleaning text with the preprocess_text function and building a custom Vocabulary class, are no longer necessary.
# The Hugging Face tokenizer handles this work for you. It performs the text cleaning, tokenization, and numerical conversion automatically inside the custom Dataset class you are about to create. You will define this Dataset to wrap your data and then use DataLoaders to create iterable batches.
# RecipeDataset Dataset Class
# You will start by defining a RecipeDataset class, the purpose of which is to use your tokenizer to convert a single raw text sample into the required numerical tensors on the fly, right when the model needs it.

# Define the RecipeDataset which will serve as a container for your data and manage the on the fly tokenization process.
# __init__: Initializes the dataset by storing your texts, labels, and the tokenizer.
# __len__: Returns the total number of samples in your dataset.
# __getitem__: This is the core method where the on the fly processing occurs. For each text sample, the single call to the tokenizer performs all the complex preprocessing steps you previously handled manually. It cleans the text, tokenizes it into sub words, converts tokens to numerical IDs using its built in vocabulary, and creates an attention mask. The method then combines these tensors with the correct label into a dictionary, ready for the model.
class RecipeDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.

    This Dataset class stores raw texts and their corresponding labels. It is
    designed to work efficiently with a Hugging Face tokenizer, performing
    tokenization on the fly for each sample when it is requested.
    """
    def __init__(self, texts, labels, tokenizer):
        """
        Initializes the RecipeDataset.

        Args:
            texts: A list of raw text strings.
            labels: A list of integer labels corresponding to the texts.
            tokenizer: A Hugging Face tokenizer instance for processing text.
        """
        # Store the list of raw text strings.
        self.texts = texts
        # Store the list of integer labels.
        self.labels = labels
        # Store the tokenizer instance that will process the text.
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return the size of the dataset based on the number of texts.
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves and processes one sample from the dataset.

        For a given index, this method fetches the corresponding text and label,
        tokenizes the text, and returns a dictionary of tensors.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the tokenized inputs ('input_ids',
            'attention_mask') and the 'labels' as tensors.
        """
        # Get the raw text and label for the specified index.
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text, handling tasks like cleaning, numerical conversion,
        # and truncation. Padding is handled later by a DataCollator.
        encoding = self.tokenizer(text, truncation=True, max_length=512)

        # Add the label to the encoding dictionary and convert it to a tensor.
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        # Return the dictionary containing all processed data for the sample.
        return encoding
    
#%% # Create the full dataset
full_dataset = RecipeDataset(texts, labels, bert_tokenizer)

#%% # Split the full dataset into an 80% training set and a 20% validation set.
train_dataset, val_dataset = helper_utils.create_dataset_splits(
    full_dataset, 
    train_split_percentage=0.8
)

# Print the number of samples in each set to verify the split.
print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

#%% Create DataLoaders
# Data collator handles dynamic padding for each batch
data_collator = transformers.DataCollatorWithPadding(tokenizer=bert_tokenizer)

#%% Create two DataLoader instances, train_loader and val_loader.
# collate_fn=data_collator: Passing your data_collator to create dynamically padded batches instead of the default PyTorch behavior.
# Set the number of samples to process in each batch.
batch_size = 32

# Create the DataLoader for the training set with `data_collator`
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=data_collator
                         )

# Create the DataLoader for the validation set with `data_collator`
val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=data_collator
                       )

#%% Training the Model
# Addressing Class Imbalance
# Calculate class weights to address the data imbalance in your training set.
# Extract all labels from the training set to calculate class weights for handling imbalance.
train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    
    
# Use scikit-learn's utility to automatically calculate class weights.
class_weights = compute_class_weight(
    # The strategy for calculating weights. 'balanced' is automatic.
    class_weight='balanced',
    # The array of unique class labels (e.g., [0, 1]).
    classes=np.unique(train_labels_list),
    # The list of all training labels, used to count class frequencies.
    y=train_labels_list
)

# Convert the NumPy array of weights into a PyTorch tensor of type float
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Print the final weights to verify the calculation.
print("Calculated Class Weights:")
print(f"  - Fruit (Class 0):     {class_weights[0]:.2f}")
print(f"  - Vegetable (Class 1): {class_weights[1]:.2f}")

#%% Configuring the Loss Function
# Define nn.CrossEntropyLoss as your loss function, and pass your previously calculated class_weights tensor to the weight parameter.
# Initialize the CrossEntropyLoss function with the calculated `class_weights`.
loss_function = nn.CrossEntropyLoss(weight=class_weights)

#%% Baseline Approach: Fine-Tuning the Entire Model
# First, you will take the standard approach: fine-tuning the entire DistilBERT model. This means that every parameter, from the initial embedding layers to the final classification layer, will have its weights updated during training. Keep in mind that we are beginning the training using the pre-trained weights and will continue to further train the model.

# This method adapts the whole model to the recipe classification task and will serve as your performance baseline. You will use the training_loop function to run the training process and see how well this approach works.

# For each batch, it explicitly unpacks the input_ids, attention_mask, and labels required by the model.
# It then fine-tunes all layers of the DistilBERT model on your dataset.
# Set the total number of epochs.
num_epochs = 3

# Call the training loop to start the full fine-tuning process.
full_finetuned_bert, full_results = helper_utils.training_loop(
    bert_model, 
    train_loader, 
    val_loader, 
    loss_function, 
    num_epochs, 
    device
)

#%% Print the validation metrics from the results_bert dictionary to review the performance of your fine-tuned model on the validation set.
# Display the results 
helper_utils.print_final_results(full_results)

#%% An Efficient Alternative: Partial Fine-Tuning
# While fine-tuning the entire model is effective, it can be computationally expensive. Now, you will explore a more efficient strategy known as partial fine-tuning. Instead of training the entire model, you will strategically freeze the majority of the model's layers and train only those most effective for adapting to the new task.

# To ensure a fair comparison between the two approaches, you must first reload the original pre-trained DistilBERT model, since the previous training loop updated the model's weights in-place.

# Note: You will see a warning that some weights were "newly initialized." This is expected. It confirms that you have successfully loaded the pre-trained DistilBERT base and attached a new, untrained classification head.
# RELOAD the base model to ensure a fair comparison
bert_model, bert_tokenizer = helper_utils.load_bert(model_path, num_classes=2)

#%% In order to perform partial fine-tuning, you need to identify the specific layers to freeze and those to train. First, start by inspecting the architecture of the DistilBERT model.
print(bert_model)

#%% The decision of which layers to freeze is based on how transformers learn hierarchically:

# Earlier Layers: The layers closer to the input learn general language features, such as grammar and basic word relationships. Since these features are useful for almost any task, they are often kept frozen. In your DistilBERT model, these are the embeddings and the first four TransformerBlock layers:
# embeddings
print("\nEmbeddings: \n")
print(bert_model.distilbert.embeddings)

# first four TransformerBlock layers
print("\nFirst four TransformerBlock layers: \n")
print(bert_model.distilbert.transformer.layer[:4])

#%% Later Layers: The layers closer to the output learn more complex and abstract features that become more specialized to the data they are trained on. These are the layers you typically want to unfreeze to adapt the model to the nuances of your new task. In your model, these are the last two TransformerBlock layers and the final classification layers:
# last two TransformerBlock layers
print("\nLast two TransformerBlock layers: \n")
print(bert_model.distilbert.transformer.layer[4:6])

# final classification layers
print("\nFinal Classifier Layer: \n")
print(bert_model.pre_classifier)
print(bert_model.classifier)

#%% 
# For the task at hand, you will unfreeze and train the final classifier head and the last two transformer layers (later layers). This allows the model to adjust its high level feature extraction to the nuances of recipe classification, while still leveraging the robust, general language understanding from its frozen layers.

# This approach tests a key hypothesis: can you achieve comparable performance to the baseline while saving significant computational resources?
# Your first step is to freeze all parameters in the model by setting their requires_grad attribute to False. This prevents their weights from being updated during the training process.
# Freeze ALL model parameters first
for param in bert_model.parameters():
    param.requires_grad = False

#%% unfreeze the last two transformer layers to make them trainable by setting their requires_grad attribute back to True.
# Unfreeze the last 2 transformer layers
# Set the number of final transformer layers to unfreeze and train.
layers_to_train = 2 

# Access the list of all transformer layers in the DistilBERT model.
transformer_layers = bert_model.distilbert.transformer.layer

# Loop backwards from the end of the layer list for the number of layers you want to train.
for i in range(layers_to_train):
    # Select a layer using negative indexing (e.g., -1 for the last, -2 for the second to last).
    layer_to_unfreeze = transformer_layers[-(i+1)]
    
    # Iterate through all parameters of the selected layer.
    for param in layer_to_unfreeze.parameters():
        # Set requires_grad to True to make the parameter trainable.
        param.requires_grad = True
        
#%% The final step is to unfreeze the model's classification head, which consists of the pre_classifier and classifier layers, to ensure it can be trained on your new task.
# Unfreeze the classifier head
# The final layers of the model must be made trainable to adapt to the new task.

# For DistilBERT, this head consists of two linear layers.
# Unfreeze the pre_classifier layer.
for param in bert_model.pre_classifier.parameters():
    param.requires_grad = True

# Unfreeze the final classifier layer.
for param in bert_model.classifier.parameters():
    param.requires_grad = True
    
#%% Training loop
# For each batch, it explicitly unpacks the input_ids, attention_mask, and labels required by the model.
# Set the total number of epochs.
num_epochs = 3

# Call the training loop to start the partial fine-tuning process.
partial_finetuned_bert, partial_results = helper_utils.training_loop(
    bert_model, 
    train_loader, 
    val_loader, 
    loss_function, 
    num_epochs, 
    device
)

# Display the results 
helper_utils.print_final_results(partial_results)

#%% Comparing Fine-Tuning Approaches
# Compare your results
helper_utils.display_results(full_results, partial_results)

#%% Testing the Fine-tuned BERT Model on New Examples
test_products = [
    "Blueberry Muffins",                  # Expected: Fruit
    "Spinach and Feta Stuffed Chicken",   # Expected: Vegetable
    "Classic Carrot Cake with Frosting",  # Expected: Vegetable
    "Tomato and Basil Bruschetta",        # Expected: Vegetable
    "Avocado Toast",                      # Expected: Fruit
    "Zucchini Bread with Walnuts",        # Expected: Vegetable
    "Lemon and Herb Roasted Chicken",     # Expected: Fruit
    "Strawberry Rhubarb Pie",             # Expected: Fruit
]

#%%  # Loop through each test product
for product in test_products:
    # Call the prediction function with the required arguments
    category = helper_utils.predict_category(
        partial_finetuned_bert, # Try it with `full_finetuned_bert` as well.
        bert_tokenizer,
        product,
        device
    )
    # Print the results
    print(f"Product: '{product}'\nPredicted: {category}.\n")
