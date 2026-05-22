#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:08:17 2026

C3 M3 L2 Building a Transformer Encoder for Text Classification

@author: ek
"""
"""
1 - Introduction
In this notebook, you'll build a Transformer Encoder from scratch and apply it to sentiment analysis—classifying movie reviews as positive or negative.

What is a Transformer Encoder?
A Transformer encoder is a neural network architecture that transforms input text into rich numerical representations. It reads an entire sequence of words at once and produces a contextualized representation for each word—meaning each word's representation contains information about how it relates to all other words in the sequence. Think of it as a sophisticated reading comprehension system that understands not just individual words, but their meanings in context.

This hands-on implementation will deepen your understanding of how encoder models work and why they've become the backbone of models like BERT, RoBERTa, and other state-of-the-art NLP systems.

What You'll Build
Starting from the attention mechanisms you've already mastered, you'll construct:

A complete encoder block with multi-head attention and feed-forward layers
A stack of encoder layers that progressively refine text representations
A classification head that uses these representations for sentiment analysis
A training pipeline that achieves strong performance on real movie reviews
By the end of this notebook, you'll have implemented the same encoder architecture that, when pre-trained on massive text corpora, becomes BERT—one of the most important breakthroughs in modern NLP.
"""
# %% 1.1 Importing Necessary Modules
import math
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import helper_utils

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %%


