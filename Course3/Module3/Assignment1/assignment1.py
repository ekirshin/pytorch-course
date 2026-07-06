#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:39:08 2026

C3 M3 A1 Building a Translation System

@author: ek
"""

# =============================================================================
# Introduction
# Welcome to the final notebook in our transformer architecture series! After exploring attention mechanisms, encoder models, and decoder models separately, we now bring everything together to build a complete Encoder-Decoder architecture for neural machine translation.
# 
# The encoder-decoder architecture is the foundation of many sequence-to-sequence (seq2seq) tasks, particularly machine translation. This architecture allows us to transform an input sequence (like an English sentence) into an output sequence (like a French translation) by learning the complex mappings between languages.
# 
# What You'll Learn
# In this notebook, we will:
# 
# Understand the complete encoder-decoder architecture and how it combines our previous components
# Implement a full transformer model for translation
# Train our model on English-to-French translation
# Evaluate and use our model to translate new sentences
# Prerequisites
# This notebook assumes you've completed the previous notebooks on:
# 
# Attention mechanisms
# Encoder architecture
# Decoder architecture
# =============================================================================
# %% 1 - Understanding the Encoder-Decoder Architecture
# =============================================================================
# The encoder-decoder architecture is a powerful framework for sequence-to-sequence learning tasks. It consists of two main components working in tandem:
# 
# The Encoder: Processes the entire input sequence and creates a rich representation (context) of it
# The Decoder: Takes this context and generates the output sequence step by step
# Think of it as a two-stage translation process:
# 
# First, the encoder "understands" the source sentence completely
# Then, the decoder "expresses" this understanding in the target language
# 1.1 How Encoder-Decoder Models Work
# The encoder-decoder architecture follows these key principles:
# 
# Encoding Phase: The encoder processes the entire input sequence (e.g., an English sentence) and produces a sequence of hidden states that capture the meaning and context of each word in relation to the entire sentence.
# 
# Context Passing: The encoder's output (hidden states) is passed to the decoder as context. In transformer models, this happens through cross-attention mechanisms.
# 
# Decoding Phase: The decoder generates the output sequence one token at a time, using:
# 
# The encoder's context (through cross-attention)
# Previously generated tokens (through self-attention)
# Learned patterns from training data
# Autoregressive Generation: During inference, the decoder generates tokens sequentially, where each new token depends on all previously generated tokens.
# 
# 1.2 Key Components and Information Flow
# The information flow in an encoder-decoder transformer can be visualized as:
# 
# Input Sequence → Encoder → Context Vectors → Decoder → Output Sequence
#      (English)              (Hidden States)              (Desired language)
# Each component plays a crucial role:
# 
# Encoder Self-Attention: Helps each word understand its context within the source sentence
# Decoder Self-Attention: Ensures coherence in the generated target sequence
# Cross-Attention: Connects source and target, allowing the decoder to "look at" relevant parts of the input when generating each output word
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import math

import numpy as np

# For data handling
from collections import Counter

# %%
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
import helper_utils

# %% 1.3 Loading the Translation Dataset
# In this assignment, you will work on a framework modular enough to be trained on datasets for different languages (but not a multi-language translator). Run the cell below to choose the language you want to work on for this assignment.
translation_pairs, target_language = helper_utils.load_dataset()

# %% 1.4 Data Preprocessing and Tokenization
# =============================================================================
# Before training a translation model, the text data needs to be properly preprocessed. This involves two key steps:
# 
# Text Normalization: Converting text to lowercase, handling special characters, and preserving language-specific features (like accents in French or umlauts in German)
# Tokenization: Splitting sentences into individual tokens (words and punctuation) while preserving contractions like "he's" or "can't"
# For this assignment, we've provided the preprocessing functions in the helper_utils module. This includes:
# 
# prepare_data(): A function that normalizes your translation pairs and creates a language-appropriate tokenizer
# MultilingualTokenizer: A tokenizer class that handles multiple languages correctly
# normalize_string(): A function that cleans text while preserving important language features
# Note: The preprocessing code is provided so you can focus on the core learning objective - building the encoder-decoder architecture. In practice, data preprocessing is crucial for model performance, but here we want you to concentrate on understanding the translation model itself.
# 
# To load and preprocess your data, run the cell below.
# 
# The function returns:
# 
# normalized_pairs: A list of cleaned (English, Target Language) translation pairs
# tokenizer: A tokenizer object you can use to convert text to tokens
# =============================================================================
normalized_pairs, tokenizer = helper_utils.prepare_data(
    translation_pairs, 
    target_language,
    max_pairs=150000,  # Process first 150,000 pairs for faster training
    max_length=40      # Keep sentences with <= 40 words
)

# Cell 4: Check some normalized pairs
import random

print(f"\nRandom normalized {target_language} pairs:")
random_samples = random.sample(normalized_pairs, min(3, len(normalized_pairs)))
for eng, target in random_samples:
    print(f"EN: {eng}")
    print(f"{target_language}: {target}")
    print("-" * 40)

# Cell 5: Use the tokenizer on custom text
custom_text = "I love programming!"
tokens = tokenizer(custom_text)
print(f"\nCustom text: {custom_text}")
print(f"Tokens: {tokens}")

# %% 2 - Building Vocabulary and Creating Data Loaders
# =============================================================================
# 2.1 Building Vocabularies
# Now you'll build vocabularies for both source (English) and target languages using the same approach as in previous notebooks.
# =============================================================================
def build_vocab(sentences, tokenizer, min_freq=1):
    """
    Build vocabulary from sentences
    """
    counter = Counter()  # Counter to count word frequencies in all sentences
    for sent in sentences:
        counter.update(tokenizer(sent))  # Tokenize sentence and add token counts
    
    # Start vocab with special tokens for translation
    # <pad>: padding token, <unk>: unknown token, <sos>: start of sequence, <eos>: end of sequence
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [w for w, c in counter.items() if c >= min_freq]
    
    # Create a mapping from word to unique index
    word2idx = {w: i for i, w in enumerate(vocab)}
    # Create a mapping from index back to word (inverse of word2idx)
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    # Return the vocab list and the two dictionaries
    return vocab, word2idx, idx2word

# %% # Extract English and target language sentences separately
eng_sentences = [eng for eng, tgt in normalized_pairs]
tgt_sentences = [tgt for eng, tgt in normalized_pairs]

# Build vocabularies for both languages
print("Building English vocabulary...")
eng_vocab, eng_word2idx, eng_idx2word = build_vocab(eng_sentences, tokenizer, min_freq=2)
print(f"English vocab size: {len(eng_vocab)}")
print(f"First 20 English vocab words: {eng_vocab[:20]}")

print(f"\nBuilding {target_language} vocabulary...")
tgt_vocab, tgt_word2idx, tgt_idx2word = build_vocab(tgt_sentences, tokenizer, min_freq=2)
print(f"{target_language} vocab size: {len(tgt_vocab)}")
print(f"First 20 {target_language} vocab words: {tgt_vocab[:20]}")

# %% 2.2 Preparing Translation Pairs
# For the encoder-decoder model, you need to prepare pairs of sequences where each source sentence is paired with its target translation. You'll add special tokens to mark the beginning and end of sequences.
def prepare_sequence(sentence, tokenizer, word2idx, max_length=20, add_special_tokens=True):
    """
    Convert a sentence to a list of indices with special tokens
    """
    tokens = tokenizer(sentence)
    
    if add_special_tokens:
        # Add <sos> at the beginning and <eos> at the end
        tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Convert tokens to indices
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        # Pad with <pad> tokens
        indices = indices + [word2idx['<pad>']] * (max_length - len(indices))
    else:
        # Truncate if too long
        indices = indices[:max_length]
    
    return indices

# %% # Prepare all translation pairs
MAX_LENGTH = 40
prepared_pairs = []

for eng, tgt in normalized_pairs:
    # Prepare source (English) - no special tokens for encoder input
    eng_tokens = tokenizer(eng)
    eng_indices = [eng_word2idx.get(token, eng_word2idx['<unk>']) for token in eng_tokens]
    
    # Pad or truncate
    if len(eng_indices) < MAX_LENGTH:
        eng_indices = eng_indices + [eng_word2idx['<pad>']] * (MAX_LENGTH - len(eng_indices))
    else:
        eng_indices = eng_indices[:MAX_LENGTH]
    
    # Prepare target - with special tokens for decoder
    tgt_indices = prepare_sequence(tgt, tokenizer, tgt_word2idx, MAX_LENGTH, add_special_tokens=True)
    
    prepared_pairs.append((eng_indices, tgt_indices))

print(f"Number of prepared pairs: {len(prepared_pairs)}")

# Show an example pair
example_idx = 0
eng_indices, tgt_indices = prepared_pairs[example_idx]

print("\nExample prepared pair:")
print(f"Original English: {normalized_pairs[example_idx][0]}")
print(f"English tokens: {[eng_idx2word[i] for i in eng_indices if i != eng_word2idx['<pad>']]}")
print(f"English indices: {eng_indices}")

print(f"\nOriginal {target_language}: {normalized_pairs[example_idx][1]}")
print(f"{target_language} tokens: {[tgt_idx2word[i] for i in tgt_indices if i != tgt_word2idx['<pad>']]}")
print(f"{target_language} indices: {tgt_indices}")

# %% 2.3 Creating PyTorch Dataset and DataLoaders
# You'll create a custom Dataset class to handle the data efficiently during training.
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation pairs
    """
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_indices, tgt_indices = self.pairs[idx]
        
        # Convert to tensors
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor
    
# %% import torch
from torch.utils.data import DataLoader, Subset

# Create the full dataset
full_dataset = TranslationDataset(prepared_pairs)

# Determine the total number of samples in the dataset
total_size = len(full_dataset)

# Set the seed for reproducibility and generate shuffled indices directly
torch.manual_seed(42)
indices = torch.randperm(total_size).tolist()

# Calculate split point
split_point = int(0.9 * total_size)

# Create train and validation indices
train_indices = indices[:split_point]
val_indices = indices[split_point:]

# Create subset datasets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

print(f"Training pairs: {len(train_dataset)}")
print(f"Validation pairs: {len(val_dataset)}")

# Create data loaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test the data loader
for src_batch, tgt_batch in train_loader:
    print(f"Source batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")
    # Show first example from batch
    print(f"\nFirst example in batch:")
    src_tokens = [eng_idx2word[idx.item()] for idx in src_batch[0] if idx.item() != eng_word2idx['<pad>']]
    tgt_tokens = [tgt_idx2word[idx.item()] for idx in tgt_batch[0] if idx.item() != tgt_word2idx['<pad>']]
    print(f"Source (English): {' '.join(src_tokens)}")
    print(f"Target ({target_language}): {' '.join(tgt_tokens)}")
    break

# %% 3 - Building the Encoder-Decoder Architecture
# 3.1 Helper Functions for Masking
# =============================================================================
# Before building the model, you need helper functions to create masks. Masks are crucial in transformer-based models to control what information the model can "see" during processing. They prevent the model from:
# 
# Paying attention to padding tokens (which are just placeholders)
# Cheating by looking at future tokens during training
# Understanding Padding Masks
# When we batch sequences together, they often have different lengths. We pad shorter sequences with zeros to make all sequences the same length:
# 
# Original sentences:
# "I am" → ['I', 'am'] → [34, 67]
# "She loves cats" → ['She', 'loves', 'cats'] → [12, 89, 45]
# 
# After padding (assuming max_length=5):
# [34, 67, 0, 0, 0]  # "I am" + padding
# [12, 89, 45, 0, 0]  # "She loves cats" + padding
# The padding mask tells the model to ignore these padding positions:
# =============================================================================
def create_padding_mask(seq, pad_idx=0):
    """
    Create a mask to hide padding tokens
    Args:
        seq: Input sequence tensor [batch_size, seq_length]
        pad_idx: Index used for padding (usually 0)
    Returns:
        Boolean mask where True = ignore this position
    """
    return (seq == pad_idx)

# %%
padded_seq = create_padding_mask(np.array([34, 67, 0, 0, 0]))
print(padded_seq)

# %% Understanding Causal Masks (Look-Ahead Masks)
# During training, the decoder generates tokens one at a time. To prevent it from "cheating" by looking at future tokens it hasn't generated yet, we use a causal mask:
def make_causal_mask(size):
    """
    Create a mask to hide future tokens (for decoder self-attention)
    Args:
        size: Sequence length
    Returns:
        Upper triangular matrix where True = ignore this position
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# %% # Example: Decoder input with padding
decoder_input = ['<sos>', 'Hello', 'world', '<pad>', '<pad>']
indices = [2, 34, 67, 0, 0]

# Create both masks
padding_mask = create_padding_mask(indices)  # [F, F, F, T, T]
subsequent_mask = make_causal_mask(5)   # Upper triangular matrix
print(subsequent_mask)

# %% 3.2 Positional Encoding
# =============================================================================
# Transformers, unlike RNNs or LSTMs, process all tokens in a sequence simultaneously through attention mechanisms. This parallel processing is powerful but comes with a limitation: the model has no inherent understanding of word order. Without position information, "The cat chased the dog" would be indistinguishable from "The dog chased the cat" to the model.
# 
# Positional encoding solves this by adding position-dependent signals to the word embeddings. These signals use sinusoidal functions with different frequencies - think of it like giving each position in the sequence a unique "signature" that the model can learn to interpret. The clever use of sine and cosine functions at different frequencies allows the model to learn relative positions (how far apart two words are) and absolute positions (where in the sentence a word appears). This positional information is simply added to the word embeddings, allowing the model to distinguish between the same word appearing in different positions.
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sinusoidal patterns.
    
    Since transformers don't have inherent notion of sequence order (unlike RNNs),
    we add positional encodings to give the model information about where each
    token appears in the sequence.
    """
    def __init__(self, max_len, d_model):
        """
        Initialize positional encoding matrix.
        
        Args:
            max_len (int): Maximum sequence length the model will handle
                          (e.g., 100 for sentences up to 100 tokens)
            d_model (int): Dimension of the model's embeddings 
                          (e.g., 256 or 512 - must match embedding size)
        
        Creates a fixed sinusoidal pattern matrix of shape [max_len, d_model]
        where each row represents the positional encoding for that position.
        """
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trained, but saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Token embeddings of shape [batch_size, seq_len, d_model]
                       where seq_len <= max_len from initialization
        
        Returns:
            Tensor: Positional encodings of shape [batch_size, seq_len, d_model]
                   (same shape as input, ready to be added to embeddings)
        
        Example:
            If x represents embeddings for "I love cats" (3 tokens):
            - Input x shape: [batch_size, 3, 256]
            - Output shape: [batch_size, 3, 256]
            - Returns positions 0, 1, 2 encoded as 256-dim vectors
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]
