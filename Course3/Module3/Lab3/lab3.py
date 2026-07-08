# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:17:02 2026

C3 M3 L3 Understanding and Building Decoder Models 

@author: ekirshin
"""
# =============================================================================
# 1. Introduction
# In this notebook, you'll explore decoder models - the generative half of the transformer architecture that powers language generation systems like GPT, ChatGPT, and other autoregressive models. While the previous notebooks covered attention mechanisms and encoders (which understand and represent text), this notebook focuses on decoders, which generate text one token at a time.
# 
# What is a Decoder?
# A decoder is a neural network architecture designed to generate sequences autoregressively - meaning it produces output tokens one by one, where each new token depends on all previously generated tokens. Think of it like autocomplete on steroids: given a starting prompt, the decoder predicts what comes next, then uses that prediction to predict what comes after that, and so on.
# 
# The key insight that makes decoders powerful is their use of self-attention with causal masking. This allows them to:
# 
# Consider all previous context when generating the next token
# Learn complex patterns and dependencies in sequential data
# Generate coherent, contextually appropriate text
# What You'll Build
# In this notebook, you'll construct a complete decoder model and train it to generate Shakespeare-style poetry. You'll see firsthand how:
# 
# Causal masking ensures the model only looks at past tokens (no "cheating" by seeing the future)
# Multi-layer architecture builds increasingly sophisticated representations of text
# Training progression transforms random output into coherent Shakespeare-inspired verse
# By the end, you'll have built the same fundamental architecture that, when scaled up, becomes GPT-3, GPT-4, and other state-of-the-art language models. The difference between your Shakespeare generator and ChatGPT is mainly scale - more data, more parameters, and more compute - but the core architecture remains the same.
# 
# Comparing with Previous Notebooks
# This builds directly on what you've learned:
# 
# Attention Notebook: You learned how attention mechanisms allow models to focus on relevant information
# Encoder Notebook: You built models that understand and represent input text
# This Decoder Notebook: You'll build models that generate text, combining attention with autoregressive generation
# The improvement you'll observe from a simple attention model to a full decoder architecture will demonstrate why modern language models use this specific design.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

import helper_utils

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# =============================================================================
# 2. What is a Decoder?
# A decoder is a neural network architecture designed for sequence generation. Unlike encoders that process entire sequences bidirectionally to create representations, decoders generate outputs autoregressively - one token at a time, using only information from previous positions.
# 
# The key insight is that decoders are fundamentally predictive: at each position, they predict what comes next based on what came before. This makes them perfect for tasks like text generation, code completion, and conversational AI.
# 
# Key Differences from Encoders
# Aspect	              Encoder	                         Decoder
# Attention	    Bidirectional (sees all)	              Causal (sees only past)
# Primary Use	Understanding	                            Generation
# Training Task	Masked Language Modeling	          Next Token Prediction
# Example Models	BERT, RoBERTa	                       GPT, LLaMA
# Output	Fixed representations	                 Variable-length sequences
# The autoregressive nature of decoders means they generate text word by word, where each new prediction becomes part of the input for generating the next token. This sequential generation process is what allows decoders to maintain coherence across long sequences.
# =============================================================================
# =============================================================================
# 3. Decoder from Scratch
# 3.1 Core Components
# A decoder consists of several key components that work together to generate sequences. You'll implement each component following the transformer architecture pattern.
# 
# Causal Masking
# The causal mask is crucial for decoder training. It ensures that when predicting a token at position i, the model can only see tokens at positions 0 to i-1, not future tokens. This prevents "cheating" during training - the model learns to predict based only on past context, just like it will have to do during generation.
# 
# For example, when processing "The cat sat", at position 2 (word "sat"), the model can see "The" and "cat" but not "sat" itself. This mask is a matrix where allowed connections are 0 and blocked connections are -infinity (which become 0 after softmax).
# 
# Visual Example for size=4:
# 
# Causal mask (True = blocked, False = allowed):
# 
# Position:     0    1    2    3
#            ┌────┬────┬────┬────┐
#         0  │ F  │ T  │ T  │ T  │  → At position 0, can only see position 0
#            ├────┼────┼────┼────┤
#         1  │ F  │ F  │ T  │ T  │  → At position 1, can see positions 0-1
#            ├────┼────┼────┼────┤
#         2  │ F  │ F  │ F  │ T  │  → At position 2, can see positions 0-2
#            ├────┼────┼────┼────┤
#         3  │ F  │ F  │ F  │ F  │  → At position 3, can see all positions
#            └────┴────┴────┴────┘
# 
# F = False (can attend)
# T = True (masked/blocked)
# During training, the decoder sees the entire target sequence at once for efficiency. Without masking, it could cheat:
# 
# Target sequence: ['', 'Hello', 'world', '']
# 
# Without mask (WRONG):
# 
# When predicting 'Hello', decoder could peek at 'world' and ''
# This doesn't match inference where it only has previously generated tokens
# With subsequent mask (CORRECT):
# 
# When predicting 'Hello', decoder can only see ''
# When predicting 'world', decoder can only see '', 'Hello'
# This matches the inference behavior
# =============================================================================
def make_causal_mask(sz):
    """
    Creates a square causal mask to prevent attention to future positions in a sequence.

    Args:
        sz (int): The size (length) of the sequence to be masked.

    Returns:
        mask (Tensor): A square upper-triangular tensor of shape [sz, sz] with 
                       negative infinity on the upper triangle and zero elsewhere.
    """
    # Initialize a square matrix filled with negative infinity
    mask = torch.full((sz, sz), float('-inf'))
    # Zero out the lower triangle and diagonal, leaving negative infinity on future positions
    mask = torch.triu(mask, diagonal=1)
    # Return the resulting attention mask
    return mask

# %% # Example: Create a causal mask for a sequence of length 5
seq_length = 5
mask = make_causal_mask(seq_length)

print("Causal mask shape:", mask.shape)
print("\nCausal mask (0 = allowed, -inf = blocked):")
print(mask)

# Demonstrate how this affects attention scores
attention_scores = torch.randn(1, seq_length, seq_length)
print("\nOriginal attention scores (random):")
print(attention_scores[0])

# Apply mask and softmax
masked_scores = attention_scores + mask
attention_weights = F.softmax(masked_scores, dim=-1)

print("\nAttention weights after masking and softmax:")
print(attention_weights[0])

# Notice: Each row can only attend to positions up to and including itself.

# %% 3.2 Positional Encoding in the Decoder
# =============================================================================
# Just like the encoder, the decoder needs positional encoding to understand word order. Without it, the attention mechanism would treat "The cat chased the dog" and "The dog chased the cat" as identical bags of words.
# 
# Decoder-Specific Considerations
# While the decoder uses the same sinusoidal positional encoding you've already seen in the encoder, its role here is even more critical:
# 
# Generation Order Matters: During text generation, the decoder produces tokens one at a time. Positional encoding helps it track where it is in the output sequence.
# 
# Cross-Attention Alignment: When the decoder attends to encoder outputs (cross-attention), positional information helps align source and target positions correctly. For example, in translation, word order often changes between languages.
# 
# Maintaining Causality: Combined with causal masking, positional encoding ensures the decoder respects the sequential nature of language generation - position 5 can't accidentally influence position 3.
# 
# The implementation remains the same - you add sinusoidal patterns to embeddings based on position, giving each token a unique "position signature" that the model uses to understand sequence structure. This positional information flows through all attention layers, helping the decoder maintain coherent sequential output generation.
# 
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding to provide sequence order information to the model.
    """
    def __init__(self, max_len, d_model):
        """
        Initializes the positional encoding module by precomputing the encoding matrix.

        Args:
            max_len (int): The maximum sequence length supported by this module.
            d_model (int): The dimensionality of the encoding vectors.
        """
        # Call the parent constructor
        super().__init__()
        # Store the maximum sequence length
        self.max_len = max_len
        # Store the model dimensionality
        self.d_model = d_model
        
        # Initialize a tensor of zeros to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        # Create a tensor representing the numerical positions in the sequence
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Calculate the scaling factor for the wavelengths of the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Assign sine transformations to the even indices of the encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Assign cosine transformations to the odd indices of the encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register the matrix as a buffer to ensure it is part of the module state but not trainable
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Retrieves the positional encodings corresponding to the input sequence length.

        Args:
            x (Tensor): The input tensor containing token embeddings.

        Returns:
            pe (Tensor): A tensor containing the positional encodings sliced to the input sequence length.
        """
        # Determine the length of the input sequence
        seq_len = x.size(1)
        # Slice the precomputed encoding matrix to match the current sequence length
        return self.pe[:, :seq_len, :]
    
# %% 3.3 Padding Mask
# As you saw in the Encoder lab, padding masks prevent the model from attending to padding tokens in batched sequences. The decoder uses padding masks in the same way - marking positions with <pad> tokens as True so they're ignored during attention computation.
def create_padding_mask(seq, pad_idx=0):
    """
    Generates a boolean mask to identify paddina tokens within a sequence.
    
    Args:
        seq (Tensor): The input sequence tensor of shape [batch_size, seq_len].
        pad_idx (int): The integer index representing the padding token.

    Returns:
        mask (Tensor): A boolean tensor of the same shape as seq, where True 
                       indicates a padding token and False indicates a real token.
    """
    # Compare each element in the sequence to the padding index to create the mask
    return seq == pad_idx

# %%  3.4 Decoder Block
# The decoder block is a single transformer decoder unit designed for autoregressive generation. It uses masked self-attention to ensure causality—each position can only attend to previous positions.
# =============================================================================
# Architecture Components
# Pre-Norm Pattern: Layer normalization is applied before each sub-layer (more stable training)
# 
# Two Sub-layers:
# 
# Masked Multi-Head Self-Attention: Builds contextual understanding from previous tokens only
# Feed-Forward Network: Position-wise transformation with ReLU activation
# Residual Connections: Original input is added after each sub-layer to preserve information flow
# 
# The block transforms input sequences while maintaining shape [batch_size, seq_len, d_model] throughout.
# =============================================================================
class DecoderBlock(nn.Module):
    """
    A single decoder block implementation for generative transformer models.
    """
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1):
        """
        Initializes the decoder block components.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            nhead (int): The number of heads in the multi-head attention mechanism.
            dim_feedforward (int): The dimension of the hidden layer in the feed-forward network.
            dropout (float): The dropout probability used across the sub-layers.
        """
        super().__init__()
        
        # First layer normalization applied before the self-attention mechanism
        self.ln1 = nn.LayerNorm(d_model)
        # Multi-head self-attention module to capture intra-sequence dependencies
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Dropout layer applied to the output of the attention mechanism
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer normalization applied before the feed-forward network
        self.ln2 = nn.LayerNorm(d_model)
        # Position-wise feed-forward network consisting of two linear transformations and a ReLU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        # Dropout layer applied to the output of the feed-forward network
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None): 
        """
        Processes the input sequence through the decoder block.

        Args:
            x (Tensor): The input sequence tensor of shape [batch_size, seq_len, d_model].
            src_mask (Tensor): An optional causal mask to prevent attention to future tokens.

        Returns:
            x (Tensor): The processed output tensor of shape [batch_size, seq_len, d_model].
        """
        # Apply layer normalization to the input prior to attention
        x_norm = self.ln1(x)
        # Perform multi-head self-attention using the normalized input as query, key, and value
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=src_mask)
        # Add the attention output to the original input via a residual connection after applying dropout
        x = x + self.dropout1(attn_out)
        
        # Apply layer normalization to the result of the first sub-layer
        ffn_in = self.ln2(x)
        # Pass the normalized data through the feed-forward network
        ffn_out = self.ffn(ffn_in)
        # Add the feed-forward output to the previous sub-layer's output via a residual connection after applying dropout
        x = x + self.dropout2(ffn_out)
        
        # Return the final processed sequence
        return x
    
# =============================================================================
# Why This Design?
# Critical Design Choices:
# 
# Causal masking: Enforces autoregressive generation - the model can't "cheat" by looking at future tokens it's supposed to predict
# 
# Pre-normalization: More stable training in deep networks compared to post-norm
# 
# Multiple dropout points: After attention and FFN outputs, creating ensemble-like regularization
# 
# Residual connections: Enable training of very deep networks (up to 96+ layers in large models)
# 
# Parameter dimensions: dim_feedforward is typically 4x d_model, balancing expressiveness with efficiency
# 
# This pattern, when stacked 6-96 times, creates the powerful generation capabilities of models like GPT. Each layer refines the representation incrementally, building from basic token patterns to complex semantic understanding.    
# =============================================================================

# %% Let's test a single decoder layer:
# Example: Process a sequence through one decoder layer
d_model = 256
nhead = 8
batch_size = 2
seq_len = 6

# Create a decoder layer
decoder_layer = DecoderBlock(d_model, nhead, dim_feedforward=1024, dropout=0.1)
decoder_layer.eval()  # Set to eval mode to disable dropout for consistent results

# Create input: batch of sequences with d_model features
input_tensor = torch.randn(batch_size, seq_len, d_model)
print(f"Input shape: {input_tensor.shape}")

# Create causal mask
causal_mask = make_causal_mask(seq_len)
print(f"Causal mask shape: {causal_mask.shape}")

# Forward pass
output = decoder_layer(input_tensor, src_mask=causal_mask)
print(f"Output shape: {output.shape}")

# The output maintains the same shape but contains refined representations
print(f"\nInput tensor stats - Mean: {input_tensor.mean():.4f}, Std: {input_tensor.std():.4f}")
print(f"Output tensor stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
print("\nNote: Layer norm keeps statistics stable across layers")

# %% 3.5 Complete Decoder Mode
# Now let's build a complete text generator using PyTorch's optimized TransformerEncoder layers configured for autoregressive generation.
# =============================================================================
# Causal Masking for Autoregressive Generation
# During training, the model processes the entire sequence at once but uses causal masking to maintain the autoregressive property:
# 
# Input: "To be or not"
# Position 0 (To):   [✓ · · ·]  → Predicts "be"
# Position 1 (be):   [✓ ✓ · ·]  → Predicts "or"  
# Position 2 (or):   [✓ ✓ ✓ ·]  → Predicts "not"
# Position 3 (not):  [✓ ✓ ✓ ✓]  → Predicts next token
# 
# ✓ = can attend to    · = masked (cannot see)
# Key Implementation Details
# TransformerEncoder as Decoder: We use PyTorch's TransformerEncoder with causal masking, which effectively creates a decoder
# Pre-Norm Architecture: norm_first=True applies layer normalization before attention (more stable)
# Dual Masking: Combines causal mask (for autoregression) and padding mask (for variable-length sequences)
# =============================================================================
class Decoder(nn.Module):
    """
    A transformer-based decoder model designed for autoregressive text generation.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, max_len=512, dropout=0.1):
        """
        Initializes the decoder model with its constituent layers and parameters.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            d_model (int): The dimensionality of the internal representation vectors.
            nhead (int): The number of heads in the multi-head attention mechanisms.
            num_layers (int): The number of transformer blocks to stack.
            dim_feedforward (int): The hidden layer dimensionality of the feed-forward networks.
            max_len (int): The maximum sequence length supported by the positional encodings.
            dropout (float): The dropout rate applied for regularization.
        """
        super().__init__()
        
        # Store the model dimensionality for use in scaling embeddings
        self.d_model = d_model
        
        # Initialize the lookup table for token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Initialize the positional encoding module to provide sequence order context
        self.pos_enc = PositionalEncoding(max_len, d_model)
        
        # Define the dropout layer for regularizing the embedding output
        self.dropout = nn.Dropout(dropout)
        
        # Define the individual transformer layer with pre-layer normalization
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True      
        )
        
        # Stack multiple transformer layers to form the deep architecture
        self.transformer_blocks = nn.TransformerEncoder(dec_layer, num_layers)
        
        # Final layer normalization to prepare the sequence for the output projection
        self.ln_final = nn.LayerNorm(d_model)
        
        # Project the high-dimensional representations back to the vocabulary space
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        """
        Performs a forward pass of the decoder on a given input sequence.

        Args:
            src (Tensor): A tensor of token indices of shape [batch_size, seq_len].

        Returns:
            logits (Tensor): A tensor containing the raw predictions for each token in the vocabulary,
                             with shape [batch_size, seq_len, vocab_size].
        """
        # Create a mask to identify and ignore padding tokens
        padding_mask = create_padding_mask(src, pad_idx=0)
        
        # Create a triangular causal mask to prevent attention to future tokens
        causal_mask = make_causal_mask(src.size(1)).to(src.device)
        
        # Convert input indices to embeddings and scale them by the square root of the dimension
        x = self.token_emb(src) * math.sqrt(self.d_model)
        
        # Add pre-computed positional encodings to the token embeddings
        x = x + self.pos_enc(x)
        
        # Apply dropout to the combined embedding and positional signal
        x = self.dropout(x)
        
        # Pass the sequence through the stacked transformer blocks using both causal and padding masks
        x = self.transformer_blocks(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Normalize the features across the final hidden dimension
        x = self.ln_final(x)
        
        # Compute the final logits for each position in the sequence
        logits = self.output_projection(x)
        
        # Return the resulting logit tensor
        return logits

# %% Let's test the complete decoder with a simple example:
# Example: Basic forward pass through decoder

vocab_size = 100  # Small vocabulary for demo
d_model = 128
nhead = 4
num_layers = 2

# Create decoder
decoder = Decoder(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=512,
    dropout=0.1
)
decoder.eval()

# Create input: batch of token indices
batch_size = 2
seq_len = 8
input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Random token IDs

print(f"Input shape: {input_ids.shape}")
print(f"Sample input: {input_ids[0]}")

# Forward pass
with torch.no_grad():
    output = decoder(input_ids)

print(f"\nOutput shape: {output.shape}")
print(f"Expected: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")

print("\n✓ Decoder transforms input tokens into vocabulary logits for next-token prediction")

# %% 4. Data Preparation
# =============================================================================
# 4.1. Download and Load Shakespeare Dataset
# We'll use Shakespeare's complete works as our training data - a perfect dataset for learning to generate poetic, dramatic text. The dataset contains plays, sonnets, and poems with rich vocabulary and distinctive style.
# =============================================================================
# Get the Shakespeare text
text = helper_utils.get_shakespeare_data()

# %% 4.2. How Decoder Training Works
# =============================================================================
# The decoder learns to predict the next token given all previous tokens. This is called autoregressive training:
# 
# Training Example:
# Input:  "To be or not to"
# Target: "be or not to be"  (shifted by 1)
# 
# The model learns:
# - Given "To" → predict "be"
# - Given "To be" → predict "or"
# - Given "To be or" → predict "not"
# - Given "To be or not" → predict "to"
# - Given "To be or not to" → predict "be"
# During training, we use teacher forcing: the model sees the real tokens as input, not its own predictions. The causal mask ensures it can't "cheat" by looking ahead.
# 
# %% 4.3. Tokenization and Vocabulary
# You'll tokenize Shakespeare's text into words and punctuation, keeping common contractions intact:
# =============================================================================
# Prepare all data components
data = helper_utils.prepare_shakespeare_data(
    text,
    vocab_size=6000,    # Top 6000 most frequent tokens
    seq_len=25,        # Sequence length for training
    batch_size=32,      # Batch size
    train_split=0.9     # 90% train, 10% validation
)

# Extract components
vocab = data['vocab']
word2idx = data['word2idx']
idx2word = data['idx2word']
train_loader = data['train_loader']
val_loader = data['val_loader']

print(f"Vocabulary size: {len(vocab)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader) if val_loader else 0}")

# %% 4.4. Training Data Structure
# =============================================================================
# Each training batch contains:
# 
# Input sequences: [batch_size, seq_len] - The text the model reads
# Target sequences: [batch_size, seq_len] - What the model should predict (input shifted by 1)
# =============================================================================
# Examine a training batch
batch = next(iter(train_loader))
inputs, targets = batch

print(f"Input shape: {inputs.shape}")
print(f"Target shape: {targets.shape}")

# Show example
idx = 0  # First sequence in batch
print(f"\nExample training pair:")
print(f"Input:  {' '.join([idx2word[i.item()] for i in inputs[idx][:10]])}...")
print(f"Target: {' '.join([idx2word[i.item()] for i in targets[idx][:10]])}...")

# %% The model will learn to generate text by predicting one token at a time, building up Shakespeare-like language patterns through many examples.
class ShakespeareGenerator(nn.Module):
    """
    A sequence generation model specialized for text synthesis.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, max_len=5000, dropout=0.1, pad_idx=0):
        """
        Initializes the generator with a decoder backbone and configuration parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embedding vectors.
            nhead (int): The number of attention heads.
            num_layers (int): The number of decoder layers.
            dim_feedforward (int): The dimension of the feedforward network.
            max_len (int): The maximum sequence length supported.
            dropout (float): The dropout probability.
            pad_idx (int): The index used for padding tokens.
        """
        super().__init__()
        
        # Initialize the core decoder architecture with the specified hyperparameters
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )
        
        # Set the padding index for the model
        self.pad_idx = pad_idx
        # Set the total vocabulary size
        self.vocab_size = vocab_size
        
    def forward(self, x):
        """
        Performs the forward pass to compute logits for the input sequence.

        Args:
            x (Tensor): Input tensor of token indices with shape [batch_size, seq_len].

        Returns:
            logits (Tensor): Output logits representing the probability distribution over the 
                             vocabulary for each position, with shape [batch_size, seq_len, vocab_size].
        """
        # Delegate the input processing to the internal decoder module
        return self.decoder(x)

# %% # Create Shakespeare generator model
model = ShakespeareGenerator(
    vocab_size=len(vocab),
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    max_len=5000,
    dropout=0.1,
    pad_idx=word2idx['<pad>']
)

# Setup training components
loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.cuda.empty_cache()  # Clear GPU memory if available

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")


# %% 5 - Training Function
# Train the model. Increase epochs (around 5) to better result, but bigger training time
EPOCHS = 5
helper_utils.train_model(model, len(vocab), train_loader, loss_fn, optimizer, epochs=EPOCHS, device=device)

# %% # Simple generation
generated = helper_utils.generate_text(
    model, 
    "To be or not",
    data['tokenizer'],
    data['word2idx'],
    data['idx2word'],
    max_length=100,
    temperature=0.3,
    device=device
)
print(generated)

# %% 6 - Conclusion
# =============================================================================
# In this lab, you built a complete decoder model from scratch with causal masking and positional encoding, then trained it to generate Shakespeare-style text using next-token prediction.
# 
# Why Decoders Excel at Text Generation:
# Causal Masking: Unlike simple attention that sees all positions, decoders enforce left-to-right generation by masking future tokens - exactly how text is naturally written.
# 
# Multi-Layer Architecture: Stacked decoder layers (vs single attention) learn hierarchical patterns from characters → words → phrases → sentences.
# 
# Purpose-Built Design: Decoders are explicitly optimized for next-token prediction, while simple attention was designed for understanding/encoding, not generation.
# 
# Results:
# 
# Simple attention: ~60-70% coherent text
# Decoder: ~85-95% coherent text with maintained style
# The decoder's specialized architecture for sequential generation makes it the foundation of modern language models like GPT, significantly outperforming attention-only approaches for text generation tasks.
# =============================================================================


