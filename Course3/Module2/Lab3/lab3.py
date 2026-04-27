#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 M2 L3 Stable Diffusion: From Image Classification to Generative Modeling

You have already mastered the art of building and training models that can see and categorize the world around them. But what if you could teach a machine not just to recognize what exists, but to create something entirely new? In this lab, you'll shift your perspective from discriminative models to the fascinating realm of generative modeling.

The core of this transformation lies in diffusion models, an elegant architectural approach inspired by the physical process of particles spreading out over time. Instead of simply predicting a label, you'll work with models that learn to reverse this process, turning random static into high-fidelity images. This hands-on experience will show you how a model can take a simple text prompt and manifest a visual reality, providing you with a clear, intuitive understanding of one of the most powerful tools in modern AI.

In this lab, you will:

Use the Stable Diffusion 2 model and the Hugging Face diffusers library to generate images from text descriptions.

Control the creative process using deterministic seeds and explore the impact of negative prompts.

Experiment with key pipeline parameters like num_inference_steps and guidance_scale to refine your results.

Visualize the denoising trajectory to see firsthand how random noise gradually evolves into a coherent image.

Let's get started!

Created on Sun Apr 26 20:51:55 2026

@author: ek
"""
import sys
import warnings

# Redirect stderr to a black hole to catch other potential messages
class BlackHole:
    def write(self, message):
        pass
    def flush(self):
        pass
sys.stderr = BlackHole()

# Ignore Python-level UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

from diffusers import StableDiffusionPipeline, DDPMPipeline
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import numpy as np

import helper_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%

