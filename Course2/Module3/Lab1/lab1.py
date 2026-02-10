#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 18:07:10 2026

@author: ek
"""

import torch
from transformers import BertTokenizerFast, AutoTokenizer

import helper_utils

#%% Manual Tokenization: Building a Vocabulary
sentences = [
    'I love my dog',
    'I love my cat'
]

#%% Implement tokenize function that converts input text to lowercase and splits it into individual words (tokens) based on whitespace.
def tokenize(text):
    """
    Tokenizes the provided text by converting it to lowercase 
    and splitting it on whitespace.

    Args:
        text: The input string to be tokenized.

    Returns:
        tokens: A list of lowercase string tokens derived from 
                the input text.
    """
    # Lowercase the text and split by whitespace
    tokens = text.lower().split()

    return tokens

#%% Implement build_vocab function that takes a list of sentences, tokenizes them, and creates a vocab (vocabulary).
# Each unique word encountered is added to the vocabulary and assigned a unique numerical ID.
def build_vocab(sentences):
    """
    Constructs a vocabulary mapping from a list of sentences, assigning 
    a unique integer ID to each unique token.

    Args:
        sentences: A list of text strings representing the corpus to 
                   be processed.

    Returns:
        vocab: A dictionary mapping unique string tokens to unique 
               integer indices.
    """
    vocab = {}

    # Iterate over the list of input sentences
    for sentence in sentences:
        # Convert the sentence into a list of tokens using the tokenizer
        tokens = tokenize(sentence)

        # Process each individual token found in the sentence
        for token in tokens:
            # Check if the token is currently missing from the vocabulary dictionary
            if token not in vocab:
                # Assign a new unique integer index to the token, starting from 1
                vocab[token] = len(vocab) + 1
    
    return vocab

#%% # Create the vocabulary index
vocab = build_vocab(sentences)

print("Vocabulary Index:", vocab, "\n")

#%% Using a Pre-trained BERT Tokenizer
sentences = [
    'I love my dog',
    'I love my cat'
]

# Define the local directory where the tokenizer is saved
local_tokenizer_path = "./bert_tokenizer_local"

# Initialize the tokenizer from the local directory
tokenizer = BertTokenizerFast.from_pretrained(local_tokenizer_path)

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

# Get the model's vocabulary (mapping from tokens to IDs)
word_index = tokenizer.get_vocab() # For BertTokenizerFast, get_vocab() returns the vocab

# Print the human-readable `tokens` for each sentence
print("Tokens:", tokens)

print("\nToken IDs:", encoded_inputs['input_ids'])

# Print unique tokens from your sentences mapped to their unique IDs 
helper_utils.print_unique_token_id_mappings(tokens, encoded_inputs['input_ids'])

#%% Using AutoTokenizer
# Define the local directory where the tokenizer is saved
local_tokenizer_path = "./bert_tokenizer_local"

# Initialize the tokenizer using the AutoTokenizer class
# This automatically loads the correct tokenizer (BertTokenizerFast in this case)
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

#%% 
sentences = [
    'I love my dog',
    'I love my cat'
]

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

# Get the model's vocabulary (mapping from tokens to IDs)
word_index = tokenizer.get_vocab() 

# Print the human-readable `tokens` for each sentence
print("Tokens:", tokens)

print("\nToken IDs:", encoded_inputs['input_ids'])

# Print unique tokens from your sentences mapped to their unique IDs 
helper_utils.print_unique_token_id_mappings(tokens, encoded_inputs['input_ids'])

#%% Try It With Your Own Sentences
sentences = [
    "My first sentence",
    "How are you today?",
    "Do you really love me?",
    "Stay away from me!"
]

#%% 
# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

# Get the model's vocabulary (mapping from tokens to IDs)
word_index = tokenizer.get_vocab()

# Print the human-readable `tokens` for each sentence
print("Tokens:", tokens)

print("\nToken IDs:", encoded_inputs['input_ids'])

# Print unique tokens from your sentences mapped to their unique IDs 
helper_utils.print_unique_token_id_mappings(tokens, encoded_inputs['input_ids'])

#%% "Out-of-Vocabulary" (OOV) Words
# A list of words that are likely "Out-of-Vocabulary" (OOV)
oov_words = ["Tokenization", "HuggingFace", "unintelligible"]

print("--- Subword Tokenization Example ---")

# Iterate through the words and show how they are tokenized
for word in oov_words:
    # The .tokenize() method is a direct way to see the subword breakdown
    subwords = tokenizer.tokenize(word)
    
    # Print the results
    print(f"Original word: '{word}'")
    print(f"Subword tokens: {subwords}\n")