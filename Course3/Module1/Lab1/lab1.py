#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M1 L1
Created on Wed Feb 25 22:07:28 2026

@author: ek
"""
import glob
import os
import random
from collections import defaultdict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import helper_utils
import training_functions

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Use Case 1: Signature Verification

# Define the path to the root directory containing the signature dataset.
signature_data_dir = './Signature_Verification_v5_v11/'