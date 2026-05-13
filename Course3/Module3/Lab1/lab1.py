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
