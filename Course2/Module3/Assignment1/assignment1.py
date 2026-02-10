# -*- coding: utf-8 -*-
"""
C2 M3 Assignment 1.
Created on Fri Feb  6 16:55:50 2026

@author: ekirshin
"""

from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers

import helper_utils

#%% 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% 
# Load the dataset and remove any rows with missing values to ensure data quality.
data_path = "./databricks-dolly-15k-dataset/databricks-dolly_augmented.csv"
df = pd.read_csv(data_path).dropna().reset_index(drop=True)

#%% # Get a list of the unique category names from the 'category' column.
unique_categories = df['category'].unique().tolist()

# Print the list of unique categories to see the initial state.
print("Initial categories in the dataset:\n")
pprint(unique_categories)

#%% Peek at the data
df.head(10)

#%% 
# In a real-world project, you often need to make strategic decisions about your target classes. The raw dataset has several granular categories. To create a more focused and effective dispatcher for this assignment, you'll consolidate similar categories. This common practice simplifies the classification task and can lead to a more robust model.

# You'll merge all question answering types into a single q_and_a target and combine information_extraction with summarization to create a new information_distillation category. This simplifies the set of routing targets your dispatcher needs to learn.

#%% 
# Create a mapping dictionary to define how categories should be merged.
# Keys are the old category names, and values are the new, consolidated names.
category_map = {
    "general_qa": "q_and_a",
    "open_qa": "q_and_a",
    "closed_qa": "q_and_a",
    "information_extraction": "information_distillation",
    "summarization": "information_distillation"
}

# Use the replace() method on the 'category' column to apply the mapping.
df['category'] = df['category'].replace(category_map)

#%% Prepare numerical labels
# Get an array of the unique, consolidated category names.
unique_categories = df['category'].unique()

# Create a dictionary to map each category name to a unique integer ID.
cat2id = {category: i for i, category in enumerate(unique_categories)}

# Create the new 'label' column in the DataFrame by applying the mapping.
# The .map() function efficiently converts each category name to its corresponding integer ID.
df['label'] = df['category'].map(cat2id)

# Create the reverse mapping from integer ID back to the category name.
id2cat = {id: category for category, id in cat2id.items()}

#%% # Extract the 'instruction' and 'label' columns into lists
texts = df['instruction'].tolist()
labels = df['label'].tolist()

#%% 
print(f"Total samples for classification: {len(texts)}\n")
print("Class distribution:")

# Iterate through the category-to-ID mapping to report on each class.
for category, label_id in cat2id.items():
    count = labels.count(label_id)
    print(f"  - label {label_id}: {category:<25} {count} samples")
    
#%% 
# Set the number of random samples to display, and random_state.
num_samples = 10
random_state = 25

# Display a sample of instruction and label pairs.
display(df[['instruction', 'label']].sample(num_samples, random_state=random_state).style.hide(axis="index"))

sample_df = df[['instruction', 'label']].sample(num_samples, random_state=random_state)

#%% 2 - Setting Up the Foundation: DistilBERT
# Download the pre-trained model and its tokenizer.
model_name="distilbert-base-uncased"
model_path="./distilbert-local-base"

# Ensure the model is downloaded
helper_utils.download_bert(model_name, model_path)

#%% Load the DistilBERT model and its tokenizer into memory, officially setting up the core of your AI dispatcher.
# Get the number of unique classes from the 'category' column.
num_classes = df['category'].nunique()

# Load the pre-trained DistilBERT model and its tokenizer from the local path.
bert_model, bert_tokenizer = helper_utils.load_bert(model_path, 
                                                    num_classes=num_classes
                                                   )

#%% 3. Assembling the Data Feed for Training
# GRADED CLASS: InstructionDataset

class InstructionDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.

    This Dataset class stores raw texts and their corresponding labels. It is
    designed to work efficiently with a Hugging Face tokenizer, performing
    tokenization on the fly for each sample when it is requested.

    Args:
        texts (list[str]): A list of raw text strings.
        labels (list[int]): A list of integer labels corresponding to the texts.
        tokenizer: A Hugging Face tokenizer instance used for processing text.
    """
    def __init__(self, texts, labels, tokenizer):
        
        ### START CODE HERE ###
        
        # Store the list of raw text strings
        self.texts = texts
        # Store the list of integer labels
        self.labels = labels
        # Store the tokenizer instance that will process the text
        self.tokenizer = tokenizer

        ### END CODE HERE ###

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves and processes one sample from the dataset.

        For a given index, this method fetches the corresponding text and label,
        tokenizes the text, and returns a dictionary of tensors.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized inputs ('input_ids',
                  'attention_mask') and the 'labels' as tensors.
        """

        ### START CODE HERE ###
        
        # Get the raw text and label for the specified index
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text. This single call handles cleaning, tokenization,
        # conversion to numerical IDs, and truncation.
        # Set `truncation` to `True` and `max_length` as 512
        encoding = self.tokenizer(text, truncation=True, max_length=512)

        # Add the label to the encoding dictionary and convert it to a tensor
        # Set `dtype` as `torch.long`
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        ### END CODE HERE ###

        return encoding
    
#%% # Initialize the dataset to verify
verify_class = InstructionDataset(texts, labels, bert_tokenizer)

print(f"Dataset initialized with {len(verify_class)} total samples.")
print("-" * 50)


# Display the first 3 instructions and their corresponding labels
print("Displaying the first 3 instructions and labels from your data:\n")
for i in range(3):
    print(f"Instruction: {verify_class.texts[i]}")
    print(f"Label:       {verify_class.labels[i]}\n")
print("-" * 50)

#%% 3.2 - Splitting Data for Training and Validation
# Create the full dataset
full_dataset = InstructionDataset(texts, labels, bert_tokenizer)


#%% # Split the full dataset into an 80% training set and a 20% validation set.
train_dataset, val_dataset = helper_utils.create_dataset_splits(
    full_dataset, 
    train_split_percentage=0.8
)

# Print the number of samples in each set to verify the split.
print(f"Training samples:    {len(train_dataset)}")
print(f"Validation samples:  {len(val_dataset)}")

#%% # GRADED FUNCTION: create_data_collator

def create_data_collator(tokenizer):
    """
    Initializes and returns a data collator for dynamic padding.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance.
    
    Returns:
        collator: A transformers.DataCollatorWithPadding instance.
    """

    ### START CODE HERE ###
    
    # Initialize the data collator
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    ### END CODE HERE ###
    
    return collator

#%% # Create a few tokenized samples with different lengths to simulate a batch
sample_1 = bert_tokenizer("This is a short sentence.")
sample_1['labels'] = torch.tensor(0)
sample_2 = bert_tokenizer("This particular sentence is quite a bit longer.")
sample_2['labels'] = torch.tensor(1)
manual_batch = [sample_1, sample_2]

# Initialize the data collator
verify_function = create_data_collator(bert_tokenizer)

# Print the original, un-padded 'input_ids' for each sample
print("--- Before Collation (Original input_ids) ---")
for i, sample in enumerate(manual_batch):
    print(f"Sample {i+1} 'input_ids': {sample['input_ids']}")

print("\n--- After Collation (Padded and batched) ---")
# Pass the list of samples to the collator and inspect the result
collated_batch = verify_function(manual_batch)

# Directly print the final tensor
print(collated_batch['input_ids'])

#%% Create the data_collator instance. This component is now ready to be plugged into the final stage of your data pipeline.
data_collator = create_data_collator(bert_tokenizer)

#%% 3.4 - Constructing the Data Loaders
# GRADED FUNCTION: create_dataloaders

def create_dataloaders(train_dataset, val_dataset, batch_size, collate_fn):
    """
    Creates and returns DataLoader instances for training and validation sets.
    
    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The number of samples per batch.
        collate_fn (callable): The function to merge a list of samples to form a mini-batch.
    
    Returns:
        tuple: A tuple containing the training DataLoader and validation DataLoader.
    """

    ### START CODE HERE ###
    
    # Create the DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    ) 
    
    # Create the DataLoader for the validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    ) 

    ### END CODE HERE ###

    return train_loader, val_loader

#%% # Create the DataLoaders
batch_size = 16
verify_function = create_dataloaders(train_dataset, val_dataset, batch_size, data_collator)

# Inspecting the Train Loader
first_train_batch = next(iter(verify_function[0]))
print("Shape of each tensor in the train batch:")
for key, value in first_train_batch.items():
    print(f"  - {key}: {value.shape}")

print()

# Inspecting the Validation Loader 
first_val_batch = next(iter(verify_function[1]))
print("Shape of each tensor in the validation batch:")
for key, value in first_val_batch.items():
    print(f"  - {key}: {value.shape}")
    
#%%  Create the final train_loader and val_loader that will fuel your model during its training.
batch_size = 32
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size, data_collator)

#%% 4 - Optimizing the Fine-Tuning Process
# 4.1 - Addressing Class Imbalance
# GRADED FUNCTION: calculate_class_weights

# GRADED FUNCTION: calculate_class_weights

def calculate_class_weights(train_dataset, device):
    """
    Calculates class weights for handling imbalanced datasets.
    
    Args:
        train_dataset (torch.utils.data.Subset): 
            The training dataset, expected to be a subset object containing 
            indices to the original dataset.
        device (torch.device): 
            The device (e.g., 'cuda' or 'cpu') to place the final tensor on.
    
    Returns:
        torch.Tensor: A 1D tensor of class weights.
    """
    # Extract all labels from the training set to calculate class weights
    train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]

    ### START CODE HERE ###

    # Use scikit-learn's utility to automatically calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_list),
        y=train_labels_list
    ) 
    
    # Convert the NumPy array of weights into a PyTorch tensor of type float
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    ### END CODE HERE ###
    
    return class_weights_tensor.to(device)

#%% # Calculate the class weights for the training set
verify_function = calculate_class_weights(train_dataset, device)

# Display the weights alongside their corresponding category
print("Breakdown of weights per category:")

# Sort the categories by their ID to ensure the order matches the weights tensor
sorted_categories = sorted(cat2id.items(), key=lambda item: item[1])

for category, label_id in sorted_categories:
    # .item() extracts the scalar value from the tensor
    weight = verify_function[label_id].item()
    print(f"  - {category:<25}: {weight:.4f}")

#%% 4.2 Configuring the Loss Function
# Calculate the class weights for the training set
class_weights = calculate_class_weights(train_dataset, device)

# Initialize the CrossEntropyLoss function with the calculated `class_weights`.
loss_function = nn.CrossEntropyLoss(weight=class_weights)

#%% 4.3 - Implementing an Efficient Fine-Tuning Strategy
# GRADED FUNCTION: partially_freeze_bert_layers

def partially_freeze_bert_layers(model, layers_to_train=3):
    """
    Freezes all but the last N transformer layers and the classification head
    of a DistilBERT model. The model is modified in-place.

    Args:
        model (transformers.DistilBertForSequenceClassification):
            The DistilBERT model to modify.
        
        layers_to_train (int):
            The number of final transformer layers to unfreeze for training.
    
    Returns:
        transformers.DistilBertForSequenceClassification:
            The same model instance, now modified with the specified layers
            unfrozen.
    """
    
    ### START CODE HERE ###
    
    # First, freeze ALL model parameters
    for param in model.parameters():
        # Freeze the parameter.
        param.requires_grad = False

    ### END CODE HERE ###
    
    # Get reference to the list of transformer layers in the model.
    transformer_layers = model.distilbert.transformer.layer

    ### START CODE HERE ###

    # Loop `layers_to_train` times to unfreeze the specified number of final layers.
    for i in range(layers_to_train):
        # Use negative indexing to select layers from the end of the list.
        layer_to_unfreeze = transformer_layers[-(i + 1)]

        # Iterate through all parameters *within the selected layer*.
        for param in layer_to_unfreeze.parameters():
            # Unfreeze the parameter.
            param.requires_grad = True

    ### END CODE HERE ###
    
    # Unfreeze the final classification head
    for param in model.pre_classifier.parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
      
    return model

#%%# Create a partially frozen model instance
verify_function = partially_freeze_bert_layers(bert_model)

# Iterate through the model's parameters to check their status
for name, param in verify_function.named_parameters():
    # Check a parameter from an early, frozen layer
    if 'layer.0.attention.q_lin.weight' in name:
        print(f"Parameter from an EARLY layer: '{name}'")
        print(f"  - Trainable (requires_grad): {param.requires_grad}")

    # Check a parameter from a later, unfrozen layer
    if 'layer.4.attention.q_lin.weight' in name:
        print(f"\nParameter from a LATE layer: '{name}'")
        print(f"  - Trainable (requires_grad): {param.requires_grad}")

    # Check a parameter from the final classification head
    if 'classifier.weight' in name:
        print(f"\nParameter from the CLASSIFIER head: '{name}'")
        print(f"  - Trainable (requires_grad): {param.requires_grad}")
        
#%% 5 - Training the Dispatcher
# Set the number of final transformer layers to unfreeze for training. 
layers_to_train = 3

# Set the learning rate for the optimizer
learning_rate = 1e-5

# Set the total number of training epochs
num_epochs = 5

# Check if the value is negative
if layers_to_train < 0:
    print(f"'layers_to_train' was set to {layers_to_train}, which is not valid.\n")
    print("The number of transformer layers to train cannot be negative. Setting layers_to_train=0")
    layers_to_train = 0

# Check if the value exceeds the total number of layers in the model
elif layers_to_train > 6:
    print(f"'layers_to_train' was set to {layers_to_train}, but DistilBERT only has 6 transformer layers.\n")
    print(f"Capping at the maximum value. Setting layers_to_train = 6")
    layers_to_train = 6

#%% Freeze layers
partial_finetune_model = partially_freeze_bert_layers(bert_model, layers_to_train)

#%% # Call the training loop to start the partial fine-tuning process.
trained_finetuned_bert, partial_results = helper_utils.training_loop(
    model=partial_finetune_model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    loss_function=loss_function, 
    learning_rate=learning_rate, 
    num_epochs=num_epochs,
    device=device
)

# Display the results 
print("Final Validation Metrics")
print(f"\nLoss:       {partial_results['val_loss']:.4f}")
print(f"Accuracy:   {partial_results['val_accuracy']:.4f}")
print(f"F1:         {partial_results['val_f1']:.4f}\n")

#%% 
helper_utils.save_training_logs(partial_results)

#%% Analyze results
helper_utils.analyze_and_plot_results(partial_results, id2cat)

#%% Testing with real text
test_examples = {
    "q_and_a": [
        {"instruction": "Explain the main stages of the water cycle.", "expected": "q_and_a"},
        {"instruction": "What is the historical significance of the Magna Carta?", "expected": "q_and_a"}
    ],
    "classification": [
        {"instruction": "Is this movie review positive or negative? 'The plot was amazing and the acting was superb!'", "expected": "classification"},
        {"instruction": "Is a tomato a fruit or a vegetable?", "expected": "classification"}
    ],
    "information_distillation": [
        {"instruction": "Summarize the main arguments of the article on climate change from this text.", "expected": "information_distillation"},
        {"instruction": "Pull out all the dates mentioned in the following project timeline.", "expected": "information_distillation"}
    ],
    "brainstorming": [
        {"instruction": "Come up with some catchy slogans for a new brand of eco-friendly sneakers.", "expected": "brainstorming"},
        {"instruction": "Suggest some team-building activities for a company retreat.", "expected": "brainstorming"}
    ],
    "creative_writing": [
        {"instruction": "Write a short story about a robot who discovers music.", "expected": "creative_writing"},
        {"instruction": "Compose a haiku about a rainy day.", "expected": "creative_writing"}
    ]
}

#%% # Set the model to evaluation mode before starting the loop.
trained_finetuned_bert.eval()

print("--- Testing Dispatcher on New Instructions ---\n")

# Loop through each category and its list of examples in the dictionary.
for category, examples in test_examples.items():
    print(f"--- Category: {category} ---")
    for example in examples:
        instruction = example["instruction"]
        expected = example["expected"]
        
        # Get the model's prediction for the instruction.
        predicted = helper_utils.predict_category(
            model=trained_finetuned_bert,
            tokenizer=bert_tokenizer,
            text=instruction,
            device=device,
            id2cat=id2cat # Pass the mapping dictionary
        )
        
        # Check if the prediction was correct and set the result string/emoji.
        if predicted == expected:
            result = "✅ Correct"
        else:
            result = "❌ Incorrect"
            
        # Print the results in the desired format.
        print(f"Instruction: '{instruction}'")
        print(f"Expected:    {expected}")
        print(f"Predicted:   {predicted}")
        print(f"Result:      {result}\n")