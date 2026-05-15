#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M3 L1 Self-Attention: Building the Foundation of Transformers

Created on Tue May 12 22:58:59 2026

@author: ek
"""
"""
Understanding Attention Through Implementation
In this notebook, you'll build self-attention from scratch to understand the mechanism that revolutionized natural language processing. Rather than treating attention as a black box, you'll see exactly how it computes relationships between words, why it's so powerful, and how it learns to focus on relevant context.

Self-attention is the core innovation that enables models like BERT, GPT, and other transformers to understand language. By the end of this notebook, you'll have implemented the same attention mechanism used in these state-of-the-art models, working through simple examples to build deep intuition about how and why it works.

Let's demystify attention by coding each component step by step, visualizing the computations, and seeing how different patterns emerge from the learning process.
"""

import math
import os
import re
import urllib.request
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# %%
# =============================================================================
# 1 - Data Preparation
# 1.1 Introduction
# Before you can train any language model, you need to convert raw text into a numerical format that the model can understand. This fundamental step transforms human-readable text into the mathematical representations that neural networks require.
# 
# You'll start by preparing a toy dataset: just a few simple sentences. This small scale lets you trace through every computation and truly understand what's happening inside the attention mechanism.
# 
# As part of this process, you'll build a vocabulary that includes:
# 
# A padding token (<pad>) for making all sequences the same length in a batch
# An unknown token (<unk>) to handle any rare or out-of-vocabulary words the model might encounter
# These special tokens are essential for handling real-world text where sequences vary in length and new words may appear during inference.
# 
# Let's begin!
# =============================================================================
# 1. Tiny toy dataset
sentences = """
the dog chased the cat
the cat chased the mouse
the dog ran fast
the mouse ran fast
the cat lay down
"""

# Build vocab
tokens = sentences.split()
vocab = ['<pad>', '<unk>'] + sorted(set(tokens))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
print("Vocab:", vocab)

# %% 1.2 The Tokenizer
# =============================================================================
# A fundamental step in any NLP pipeline is tokenization. A tokenizer splits raw text into meaningful chunks (tokens)—typically words—that become the basic units your model processes. Since neural networks work with numbers, not text, the tokenizer also handles the crucial task of mapping these tokens to unique numerical IDs.
# 
# Below, you'll define a simple Python tokenizer as a class. This version uses regular expressions to extract whole words while normalizing the text by:
# 
# Converting everything to lowercase for consistency
# Removing punctuation to focus on word meanings
# Splitting on whitespace and word boundaries
# This clean, simplified approach is perfect for understanding attention mechanics without getting distracted by complex linguistic edge cases.
# 
# =============================================================================
class SimpleTokenizer:
    """
    Splits on whitespace and lowercases, with optional regex for real word tokens.
    """
    def __init__(self):
        """
        Initializes the SimpleTokenizer instance.
        """
        pass

    def __call__(self, text):
        """
        Processes the input text into a list of lowercase word tokens.
        
        Args:
            text: The string to be tokenized.
            
        Returns:
            A list of strings containing the extracted tokens.
        """
        # Option 1: Basic split (uncomment to just split on spaces)
        # return text.lower().split()
        # Option 2: More robust - returns only word tokens (ignores punctuation)
        return re.findall(r'\b\w+\b', text.lower())

# %%
# Usage example:
tokenizer = SimpleTokenizer()
tokens = tokenizer("The Dog chased the Cat.")
print(tokens)  # Output: ['the', 'dog', 'chased', 'the', 'cat']

# %% 1.3 Building the Vocabulary
# =============================================================================
# Once you have a tokenizer that splits your sentences into individual words, the next step is to build a vocabulary—a list of all unique tokens that appear in your dataset.
# 
# This vocabulary is crucial because it links each word to a unique ID, which your model will use throughout training and inference. You'll also want to include two special tokens in your vocabulary: one for padding (<pad>) and one for unknown or out-of-vocabulary tokens (<unk>).
# 
# Below you'll find a function to create:
# 
# the vocabulary list,
# a dictionary that maps each token to its unique index (word2idx),
# and a reverse mapping from indices to words (idx2word).
# You can adjust the min_freq parameter to exclude rare words and keep your vocabulary more manageable.
# =============================================================================
def build_vocab(sentences, tokenizer, min_freq=1):
    """
    Constructs a vocabulary and mapping dictionaries from a collection of sentences.

    Args:
        sentences: A collection of strings to be processed.
        tokenizer: A callable object used to split sentences into individual tokens.
        min_freq: The minimum number of occurrences required for a token to be 
                  included in the vocabulary.

    Returns:
        vocab: A list of unique tokens including special padding and unknown markers.
        word2idx: A dictionary mapping each token string to its unique integer index.
        idx2word: A dictionary mapping each integer index back to its token string.
    """
    # Initialize a frequency counter for tokens
    counter = Counter()
    # Iterate through each sentence to update token counts
    for sent in sentences:
        # Generate tokens using the provided tokenizer and update the counter
        counter.update(tokenizer(sent))

    # Define the initial vocabulary with special tokens and filter by frequency
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]

    # Map each unique word in the vocabulary to a specific integer index
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Map each integer index back to its corresponding word
    idx2word = {i: w for i, w in enumerate(vocab)}

    # Provide the list of tokens and the bidirectional mapping dictionaries
    return vocab, word2idx, idx2word

# %% # Using our sample sentences and tokenizer
sentences = [
    "the dog chased the cat",
    "the cat chased the mouse",
    "the dog ran fast",
    "the mouse ran fast",
    "the cat lay down"
]

tokenizer = SimpleTokenizer()                 # Define the tokenizer (splits into lowercase words)
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)  # Build vocab & mappings

print(vocab)

# %%
word = 'dog'
id_word = word2idx[word]
print(f"ID for word = {word}: {id_word}")
print(f"Word for ID = {id_word}: {idx2word[id_word]}")

# %% 1.4 Sliding Windows
# =============================================================================
# Now that you have a list of tokenized and numericalized sentences, you need to organize your data into (input, target) pairs suitable for training a language model.
# 
# The most common approach for this is to use a sliding window: you select a fixed-size window of words as input, and train the model to predict the next word that follows the window. By moving this window across every sentence in your dataset, you generate many (input, target) examples for training.
# 
# This method helps your model learn the sequential structure of language, so it can anticipate what comes next given a context.'
# Now you'll implement this sliding window approach in code. This will prepare your input and target lists, ready for training your attention-based model.
# =============================================================================
tokenizer = SimpleTokenizer()
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)

# 2. Parameters
SEQ_LEN = 4   # Length of input sequence for each example

# 3. Convert sentences to token ID lists
encoded_sentences = []  # Will be a list of lists of token IDs
for sent in sentences:
    tokens = tokenizer(sent)  # Split sentence into tokens
    ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]  # Map tokens to IDs
    encoded_sentences.append(ids)

# 4. Create sliding window dataset (inputs, targets)
inputs = []
targets = []
for ids in encoded_sentences:
    # For each possible window in the sentence
    for i in range(len(ids) - SEQ_LEN):
        window = ids[i:i+SEQ_LEN]        # Input: SEQ_LEN-token window
        target = ids[i+SEQ_LEN]          # Target: next token after the window
        inputs.append(window)
        targets.append(target)

# 5. Let's show the dataset as text for illustration
for inp, tgt in zip(inputs, targets):
    inp_words = [idx2word[i] for i in inp]
    tgt_word = idx2word[tgt]
    print(f"Input: {inp_words}  →  Target: {tgt_word}")

# %% 1.5 Turning the Data Into a PyTorch Dataset
# =============================================================================
# Now that you’ve prepared your inputs and targets lists, it’s time to convert them into a format that PyTorch can use for training: a Dataset.
# 
# A PyTorch Dataset is a simple class structure that knows how to return an input–target pair given an index. This makes it easy to batch your data and feed it into a model, especially for larger sets of examples.
# 
# Below, you’ll see how to wrap your lists in a Dataset, and then use a DataLoader to shuffle and batch your data automatically during training.
# =============================================================================
class TinyDataset(Dataset):
    """
    A custom Dataset class for managing input-target pairs in a PyTorch-compatible format.

    Args:
        inputs: A list or array of input sequences/windows.
        targets: A list or array of target labels corresponding to the inputs.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset by converting raw data into tensors.

        Args:
            inputs: The input data sequences.
            targets: The target labels or word indices.
        """
        # Convert input windows to a long tensor for efficient indexing
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        # Convert target indices to a long tensor
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        """
        Provides the total number of samples available in the dataset.

        Returns:
            An integer representing the total count of input samples.
        """
        # Return the total length of the input tensor
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample and its corresponding target by index.

        Args:
            idx: The integer index of the desired sample.

        Returns:
            sample: The input tensor at the specified index.
            target: The target tensor at the specified index.
        """
        # Return the input-target pair as a tuple for the requested index
        return self.inputs[idx], self.targets[idx]
    
# %%
# Create an instance of your dataset
dataset = TinyDataset(inputs, targets)

# Create a DataLoader that will feed batches of data to your model during training.
# batch_size=4: each batch will contain 4 (input, target) pairs
# shuffle=True: randomize order each epoch to improve training
# num_workers=0: no extra processes for loading data (good for small datasets)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)    

# %% 2 - Manual Self-Attention
# =============================================================================
# 2.1 Introduction
# Now you'll dig into the heart of what makes Transformers powerful: self-attention.
# 
# What is Self-Attention?
# Self-attention is a mechanism that allows each word in a sequence to "look at" and "gather information from" every other word, including itself. Unlike traditional sequential models that process words one by one, self-attention computes relationships between all words simultaneously.
# 
# Think of it this way: when you read the sentence "The cat sat on the mat because it was tired," your brain automatically knows that "it" refers to "cat." Self-attention gives neural networks this same capability to understand relationships and dependencies between words, no matter how far apart they are.
# 
# The Self-Attention Algorithm
# Here's how self-attention transforms a sequence:
# 
# Input: "The cat sat"
#          ↓
#    [Embeddings]
#          ↓
#    Q, K, V Projections
#          ↓
#    Attention Scores (Q × K^T)
#          ↓
#    Attention Weights (Softmax)
#          ↓
#    Weighted Sum (Weights × V)
#          ↓
# Output: Context-aware representations
# The Core Components
# Self-attention operates through three learned transformations of your input:
# 
# Queries (Q): "What information am I looking for?"
# 
# Each word generates a query vector representing what it wants to know
# Keys (K): "What information do I contain?"
# 
# Each word generates a key vector advertising what information it has
# Values (V): "What information will I actually provide?"
# 
# Each word generates a value vector containing its actual contribution
# The Attention Computation Flow
# Step 1: Linear Projections
# Input → W_Q → Queries
# Input → W_K → Keys  
# Input → W_V → Values
# 
# Step 2: Similarity Scores
# Scores = Q × K^T / √d_k
# (How relevant is each word to every other word?)
# 
# Step 3: Attention Weights  
# Weights = Softmax(Scores)
# (Convert scores to probabilities)
# 
# Step 4: Weighted Aggregation
# Output = Weights × V
# (Combine values based on attention weights)
# Visual Example
# Consider the sentence "The cat sat":
# 
# Attention Weight Matrix:
#         The   cat   sat
# The   [0.6   0.3   0.1]  ← "The" pays most attention to itself
# cat   [0.2   0.7   0.1]  ← "cat" focuses mainly on itself  
# sat   [0.1   0.3   0.6]  ← "sat" attends to "cat" and itself
# 
# Each row shows where that word "looks" for information
# After training, these attention patterns become more sophisticated:
# 
# "sat" might learn to strongly attend to "cat" (the subject)
# Articles like "the" might distribute attention more evenly
# Pronouns would learn to attend to their antecedents
# Why Self-Attention is Powerful
# Parallel Processing: Unlike RNNs, all positions are processed simultaneously
# Long-Range Dependencies: Can directly connect distant words without sequential steps
# Interpretability: Attention weights show what the model is "looking at"
# Flexibility: Learns task-specific attention patterns automatically
# Step by step, you'll implement each part of the self-attention formula—projections, dot products, scaling, softmax, and the final weighted sum—all by hand. This careful, explicit walkthrough will help you truly understand how self-attention works under the hood. Start slow here; later on, you'll see how these same ideas power much more complex, real-world NLP models!
# =============================================================================

class ManualSelfAttention(nn.Module):
    """
    A custom PyTorch module that implements a standard self-attention mechanism.
    
    Args:
        d: The dimensionality of the input and output features.
    """
    def __init__(self, d):
        """
        Initializes the linear layers for query, key, and value projections.

        Args:
            d: The dimensionality of the hidden state.
        """
        super().__init__()
        # Define the linear transformation for query vectors
        self.to_q = nn.Linear(d, d, bias=False)
        # Define the linear transformation for key vectors
        self.to_k = nn.Linear(d, d, bias=False)
        # Define the linear transformation for value vectors
        self.to_v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        """
        Executes the forward pass of the self-attention mechanism.

        Args:
            x: The input tensor of shape [batch, sequence_length, dimension].

        Returns:
            out: The resulting context-aware representations after attention weighting.
            attn: The attention weight matrix representing token interactions.
        """
        # Project the input tensor into query space
        Q = self.to_q(x)
        # Project the input tensor into key space
        K = self.to_k(x)
        # Project the input tensor into value space
        V = self.to_v(x)

        # Calculate raw attention scores using scaled dot-product between queries and keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        # Apply softmax to normalize scores into probability distributions
        attn = F.softmax(scores, dim=-1)

        # Aggregate the value vectors based on the computed attention weights
        out = torch.matmul(attn, V)

        # Provide the transformed sequence and the attention weights
        return out, attn
    
# %% 
sentence = "the dog chased the cat"
tokens = tokenizer(sentence)
print("Tokens:", tokens)  # ['dog', 'chased', 'cat']

token_ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]
print("Token IDs:", token_ids)  # e.g. [2, 3, 4]
    
embedding_dim = 2
# We'll make trainable embeddings for realism:
embed = nn.Embedding(len(vocab), embedding_dim)
torch.manual_seed(42)  # For reproducibility
x = embed(torch.tensor(token_ids).unsqueeze(0))  # shape: (1, 5, 2)
print("Input embeddings:\n", x)

# %%
attn_layer = ManualSelfAttention(embedding_dim)

# Put the input through self-attention
out, attn = attn_layer(x)

print("Attention weights:\n", attn[0].detach().numpy())
print("Output representations:\n", out[0].detach().numpy())

# %%
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("idx2word:", idx2word)

print("\nAttention Weights Matrix (rows: query token, columns: attended token):")
for i, w in enumerate(tokens):
    row = ["{:.2f}".format(a) for a in attn[0, i].detach().cpu().numpy()]
    print(f"{w:>8} attends to -> {row}")

# %% 2.2 Self-Attention with Position Embeddings
# =============================================================================
# Attention is Position-Blind
# Self-attention lets each word "look at" every other word in the sequence—but on its own, it treats the input like an unordered set of words. The attention mechanism itself has no inherent notion of sequence or position. Consider these two sentences:
# 
# "The cat chased the dog"
# "The dog chased the cat"
# Without position information, self-attention would produce identical representations for both sentences since they contain the same words! This is clearly wrong—word order fundamentally changes meaning.
# 
# The Solution: Adding Position Information
# To help the model distinguish word order, you give it a positional embedding: a unique vector for each position in the input sequence. These position vectors act like GPS coordinates for words, telling the model not just what each word is, but where it appears in the sequence:
# 
# Word Embedding:     [cat] → [0.2, 0.5, -0.1, ...]  (what the word means)
# Position Embedding: [pos 2] → [0.1, -0.3, 0.4, ...]  (where it appears)
#                            ↓
# Combined Input:     [0.3, 0.2, 0.3, ...]  (meaning + position)
# By adding position embeddings to word embeddings, you create position-aware representations that allow self-attention to learn different patterns based on word order.
# 
# Learned vs. Sinusoidal Positional Encodings
# Note: In this notebook, you're using learned positional embeddings (nn.Embedding) - this is the approach shown in the course lectures. The model learns the best position representations during training, just like it learns word embeddings.
# 
# However, in future notebooks (starting with the Encoder lab), you'll see an alternative approach: sinusoidal positional encodings. Instead of learning position embeddings, sinusoidal encodings use fixed mathematical functions (sine and cosine waves at different frequencies) to represent positions.
# 
# Both approaches work well, but they have different trade-offs:
# 
# Learned Embeddings (used in this lab):
# 
# Flexible and can adapt to your specific data patterns
# Simple to understand and implement
# Limited to the maximum sequence length seen during training
# Require max_len × d_model parameters
# Sinusoidal Encodings (used in later labs):
# 
# Fixed mathematical pattern that extends infinitely—can handle sequences longer than seen during training
# Zero learnable parameters—the encodings are completely deterministic
# Enable the model to easily learn relative positions (e.g., "3 words to the left")
# Different frequencies capture both fine-grained (nearby words) and coarse-grained (distant words) relationships
# Have become the standard in modern transformer implementations
# For this introduction to attention, learned embeddings are simpler to understand and work perfectly fine for our fixed-length sequences. You'll explore sinusoidal encodings in depth in the next lab.
# 
# The Complete Architecture
# The code below implements a complete position-aware attention model with four key components:
# 
# Token Embeddings (tok_embed): Converts token IDs to semantic vectors capturing word meaning
# Position Embeddings (pos_embed): Learnable vectors for each position (1st word, 2nd word, etc.)
# Self-Attention (attn): Computes context-aware representations using the manual attention from earlier
# Output Projection (fc): Maps the final hidden state to vocabulary size for next-word prediction
# The forward pass:
# 
# Generates position indices (0, 1, 2, ...) for each token in the batch
# Looks up both word and position embeddings
# Sums them together (this is how position information gets injected)
# Applies self-attention to these position-aware embeddings
# Takes the last position's output to predict the next word
# Projects to vocabulary size to get prediction scores
# This architecture demonstrates the minimal components needed for a position-aware language model—the foundation that, when scaled up with multiple layers and heads, becomes the transformer models powering modern NLP.
# 
# =============================================================================
class SelfAttnWithPositionalEmbedding(nn.Module):
    """
    A neural network module that combines token and positional embeddings with self-attention.

    Args:
        vocab_size: The total number of unique tokens in the vocabulary.
        seq_len: The fixed length of input sequences.
        emb_dim: The dimensionality of the embedding vectors.
    """
    def __init__(self, vocab_size, seq_len, emb_dim):
        """
        Initializes the embedding layers, attention mechanism, and output projection.

        Args:
            vocab_size: Size of the vocabulary.
            seq_len: Maximum length of the sequence.
            emb_dim: Dimension of the embeddings.
        """
        super().__init__()
        # Table for mapping token indices to continuous vector representations
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        # Learnable table for mapping sequence positions to vector representations
        self.pos_embed = nn.Embedding(seq_len, emb_dim)
        # Self-attention module for capturing dependencies between tokens
        self.attn = ManualSelfAttention(emb_dim)
        # Linear layer to project context-aware representations back to vocabulary size
        self.fc = nn.Linear(emb_dim, vocab_size)
        # Internal storage for the maximum sequence length
        self.seq_len = seq_len

    def forward(self, token_ids):
        """
        Processes input token sequences to produce prediction logits and attention weights.

        Args:
            token_ids: A tensor of token indices with shape [batch_size, seq_len].

        Returns:
            logits: Predicted scores for the next token based on the final hidden state.
            attn_weights: The attention weight matrix from the self-attention layer.
        """
        batch_size, seq_len = token_ids.shape
        # Generate a range of indices representing the position of each token in the sequence
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Retrieve word-level embedding vectors for the input tokens
        word_vecs = self.tok_embed(token_ids)
        # Retrieve learnable positional embedding vectors for each sequence index
        pos_vecs = self.pos_embed(positions)
        # Combine word and position information via element-wise summation
        input_vecs = word_vecs + pos_vecs
        # Pass the combined embeddings through the self-attention mechanism
        attn_out, attn_weights = self.attn(input_vecs)
        # Extract the hidden representation corresponding to the final token in the sequence
        last_hidden = attn_out[:, -1, :]
        # Transform the final hidden state into prediction logits over the vocabulary
        logits = self.fc(last_hidden)
        # Provide the prediction scores and the attention weights for evaluation or analysis
        return logits, attn_weights
    
# %%
x = torch.tensor([[[ 0.12, -0.55,  0.33,  0.10],
                   [-0.44,  0.91, -0.12, -0.77],
                   [ 0.48,  0.02,  0.05,  0.39],
                   [ 0.12, -0.55,  0.33,  0.10],
                   [-0.30,  0.14, -0.70,  0.81]]])  # [1, 5, 4]

attn = ManualSelfAttention(d=4)
out, attn_weights = attn(x)

print("Attention weights matrix (attn_weights):\n", attn_weights[0].detach().numpy())
print('\nExplanation:')
print("Each row i shows the attention distribution (softmaxed) over all positions in the input sequence,")
print("when computing the updated representation for token i. Rows sum to 1.\n")

print("Self-attention output (out):\n", out[0].detach().numpy())
print('\nExplanation:')
print("Each row is the new vector for input position i, computed as a weighted sum")
print("of the original value vectors, using that row from the attention weights matrix as weights.\n")   

# %% 2.3 A Practical Example
# =============================================================================
# Now you’ll see self-attention with positional encoding in action!
# In this section, you’ll:
# 
# Instantiate your custom self-attention model,
# Select a real sample from your dataset,
# Run the model to obtain predictions and attention maps,
# Visualize exactly how the model "pays attention" to different words in the input.
# The code below walks you through these steps. You can run it before and after training to see how the model's focus changes as it learns.
# =============================================================================
# 1. Set up model parameters and create your attention model
vocab_size = len(vocab)
embed_dim = 8         # Number of embedding dimensions (try 4, 8, 16, etc.)
seq_len = SEQ_LEN     # Length of your training window
model = SelfAttnWithPositionalEmbedding(vocab_size, seq_len, embed_dim)

# %%
def plot_attention(attn_weights, tokens, title="Self-Attention Map"):
    """
    Visualizes the self-attention weights for a sequence using a heatmap.
    
    Args:
        attn_weights: A tensor of attention scores, usually of shape [batch, seq_len, seq_len].
        tokens: A list of strings representing the tokens for labeling the axes.
        title: A string to be used as the title of the generated plot.

    Returns:
        None. This function displays a plot directly.
    """
    # Extract attention weights for the first batch entry and transfer to host memory
    aw = attn_weights[0].detach().cpu().numpy()
    # Initialize the figure with dimensions scaled to the number of tokens
    plt.figure(figsize=(1.2 * len(tokens), 5))
    # Render the attention matrix as a heatmap with a blue color gradient
    plt.imshow(aw, cmap='Blues')
    # Assign token strings to the horizontal axis with a specific rotation for readability
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    # Assign token strings to the vertical axis
    plt.yticks(range(len(tokens)), tokens)
    # Include a legend showing the mapping of colors to attention intensity
    plt.colorbar()
    # Set the provided title for the visualization
    plt.title(title)
    # Adjust layout parameters to prevent label clipping
    plt.tight_layout()
    # Render the final visualization to the screen
    plt.show()
