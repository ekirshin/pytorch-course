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
def load_model_pipeline(model_id, pipeline_class, device=None, **kwargs):
    """
    Loads a Hugging Face pipeline with caching logic.
    
    Args:
        model_id (str): The repository ID (e.g., "stabilityai/stable-diffusion-2-base").
        pipeline_class (class): The class to load (e.g., StableDiffusionPipeline).
        device (torch.device, optional): Device to move the pipeline to.
        **kwargs: Additional arguments for from_pretrained (e.g., torch_dtype, variant).
    
    Returns:
        The loaded pipeline.
    """
    # Define the directory for caching model files
    cache_dir = "./models"
    
    # Construct the specific cache directory name based on the model ID
    model_cache_dir_name = f"models--{model_id.replace('/', '--')}"
    
    # Create the full path to the model cache
    model_cache_path = cache_dir + "/" + model_cache_dir_name

    # Verify model snapshot integrity or perform extraction if missing
    helper_utils.check_model_snapshot(model_cache_path)

    # Instantiate the pipeline with local files and cache settings
    pipe = pipeline_class.from_pretrained(
        pretrained_model_name_or_path=model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        **kwargs  # Forward additional configuration arguments
    )

    # Transfer the pipeline to the specified device if one is provided
    if device:
        pipe = pipe.to(device)

    return pipe

# %% Initializing Stable Diffusion 2
#  Define the model_id pointing to the Stable Diffusion 2 repository.
model_id = "stabilityai/stable-diffusion-2-base"

# %%
# =============================================================================
# torch_dtype=torch.float16: Reduces memory footprint by using half precision.
# variant="fp16": Loads specific weights optimized for efficiency.
# =============================================================================
# Load the pipeline using the helper function
pipe = load_model_pipeline(
    model_id=model_id,                       # The specific model repository ID (Stable Diffusion 2)
    pipeline_class=StableDiffusionPipeline,  # The class definition for the pipeline you want to load
    device=device,                           # Moves the pipeline to the detected hardware (CUDA/MPS/CPU)
    # Extra arguments specific to this model:
    torch_dtype=torch.float16,               # Uses half-precision (float16) to reduce memory usage
    variant="fp16"                           # Loads the specific fp16 weights file for efficiency
)

print("\nLoading Complete!")

# %% Controlling Randomness: PyTorch Generators
# Stable Diffusion always starts from a state of total randomness, but you have the power to control this randomness through a seed.
# Set the seed for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# %% The Dynamics of Random Seeds
# If you call the pipeline multiple times without resetting your generator, you will get different results because you are effectively using different sections of the random sequence. To ensure your work is reproducible and you get the exact same image every time, you'll need to re-initialize or re-seed your generator before each call.
# Define a text prompt to guide the model's creative process.
# prompt = "A skeleton walking on a beach."
prompt = "A skunk riding a surf board."

# %% Run your prompt through the pipeline. You'll use 40 inference steps to balance generation speed with visual detail.
# Using 40 steps provides a good balance between quality and speed
# (Typical range: 20-50 for fast generation, 50-150 for high quality)
images = pipe(
    prompt,                 # What you want the model to create
    num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
    generator=generator     # Ensures reproducible noise/randomness
).images

helper_utils.display_images(images[0])

# %%
# Run the pipeline again using the same generator object without resetting it. You will see that even though the seed was initially 42, the image changes because the generator has moved forward.
images = pipe(
    prompt,                 # What you want the model to create
    num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
    generator=generator     # Ensures reproducible noise/randomness
).images

helper_utils.display_images(images[0])

# %% To achieve a truly deterministic result across repeated runs, you must reset the generator state before every call. Run the next cells to verify how re-initializing the generator locks in the same visual outcome.
# First run
generator = torch.Generator(device=device).manual_seed(42)

images = pipe(
    prompt,                 # What you want the model to create
    num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
    generator=generator     # Ensures reproducible noise/randomness
).images

helper_utils.display_images(images[0])

# %% # Second run
generator = torch.Generator(device=device).manual_seed(42)
images = pipe(
    prompt,                 # What you want the model to create
    num_inference_steps=40, # How many denoising steps to use (more steps = more detail/compute)
    generator=generator     # Ensures reproducible noise/randomness
).images

helper_utils.display_images(images[0])

# %% Understanding Pipeline Outputs
# =============================================================================
# You'll notice we access the result with .images[0]. This is because diffusion pipelines are designed to handle batching, allowing you to generate multiple variations of a prompt simultaneously. Even when you only request a single image, the pipeline returns a list by default.
# 
# Use num_images_per_prompt to generate three distinct images in one go.
# =============================================================================
image_list = pipe(
    prompt,                    # What you want the model to create
    num_inference_steps=40,    # How many denoising steps to use (more steps = more detail/compute)
    generator=generator,       # Ensures reproducible noise/randomness
    num_images_per_prompt = 3  # Number of images to generate
).images

helper_utils.display_images(image_list)

# %% Tuning Your Creative Engine: Pipeline Parameters
#  Refining Through Iteration: num_inference_steps
# This parameter determines how many times the model refines the noise. Increasing this number generally leads to more detailed textures and sharper edges, but it also increases the computational time required for each generation.
prompt = "a cute dog with a red bandana, sitting in a lush park"
#prompt = "a cute raccoon elegantly dressed in a suit, sitting in a restaurant at a table and ejoying its time with a beer"
# prompt = "cute 3 bears riding bicycles"

# %%
# Redefining the generator
generator = torch.Generator(device=device).manual_seed(39)

# Fewer steps (fast, less detail)
image_fast = pipe(prompt, num_inference_steps=10, generator=generator).images[0]

# Default/standard (balanced quality)
image_standard = pipe(prompt, num_inference_steps=50, generator=generator).images[0]

# More steps (slower, more detail)
image_high_quality = pipe(prompt, num_inference_steps=200, generator=generator).images[0]

images = [image_fast, image_standard, image_high_quality]
titles = ["10 steps (fast)", "50 steps (standard)", "200 steps (high quality)"]

helper_utils.display_images(images, titles=titles)

# %% Directing the Vision: guidance_scale
# =============================================================================
# Also known as Classifier-Free Guidance (CFG), this setting controls how strictly the model adheres to your text description.
# 
# Low Values (5-7): These grant the model more creative liberty, often leading to more artistic or varied results that might wander slightly from the prompt.
# 
# Standard Values (~7.5): This is the typical default, providing a solid balance between following your instructions and maintaining high visual quality.
# 
# High Values (10-15): These force a more literal interpretation of your words. However, pushing this too high can result in over-saturated or unnatural-looking artifacts.
# 
# Think of guidance_scale as a dial for the model's "imagination." Higher settings turn down the imagination in favor of literal obedience.
# =============================================================================
prompt = ("A surreal landscape with floating clocks, melting trees, "
          "and a purple sky, in the style of Salvador Dalí")
guidance_scales = [5, 7.5, 12]
# Generate all images first (so the generator seed doesn't advance unpredictably)
images = []
for gs in guidance_scales:
    # It's important to re-create the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)
    print(f"Generating image for guidance scale = {gs}")
    img = pipe(prompt, generator=generator, guidance_scale=gs).images[0]
    images.append(img)
    # Adding line space
    print()

helper_utils.display_images(
    images, 
    titles=[f"guidance_scale = {gs}" for gs in guidance_scales]
)
# %% Shaping Through Exclusion: negative_prompt
# =============================================================================
# Sometimes, describing what you don't want is just as essential as describing what you do want. The negative_prompt allows you to list attributes, colors, or objects that the model should actively avoid.
# 
# This is a fundamental technique for improving image quality. You can use it to steer the model away from common issues like blurry textures or anatomical distortions, or simply to exclude specific elements that keep appearing in your results.
# =============================================================================
prompt = "A group of realistic teddy bears eating pizza at a birthday party"
negative_prompt = "pepperoni, deformed hands, extra limbs, blurry, out of focus, text, watermark"

# %%
# Run a comparison to see how the addition of a negative_prompt cleans up the generated scene, specifically removing the pepperoni and focusing the details.
# Generate without negative prompt
generator = torch.Generator(device=device).manual_seed(42)
result_no_neg = pipe(
    prompt, 
    generator=generator, 
    num_inference_steps=30
)

# %% # Re-seed to ensure the same starting noise for fair comparison
generator = torch.Generator(device=device).manual_seed(42)
result_with_neg = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    generator=generator, 
    num_inference_steps=30
)

# %% 
imgs = [result_no_neg.images[0], result_with_neg.images[0]]
titles = ['Without negative_prompt', 'With negative_prompt']

helper_utils.display_images(imgs, titles=titles)

# %% Peeking Into the Process: Intermediate Denoising Steps
# =============================================================================
# In many research and creative workflows, you'll find it incredibly useful to inspect the intermediate states of an image during its generation. This allows you to visualize the denoising trajectory, helping you understand how your text prompt is gradually manifested from randomness into structure.
# 
# The Hugging Face diffusers library provides a flexible callback mechanism that allows you to capture the state of the image after every single step of the process.
# =============================================================================

# =============================================================================
# Capturing the Latent Evolution
# The diffusion pipeline allows you to supply a callback function that runs periodically during generation. This function receives information about the current step, the specific timestep, and the latents—the compact, mathematical representation that the model is actually refining.
# 
# You can use this callback to save these individual snapshots, decode them from their mathematical latent space into standard RGB pixels using the model's Variational Autoencoder (VAE), and observe the visual journey.
# 
# The save_intermediate_steps Workflow
# Latent Rescaling: The function first scales the raw latents by a standard factor expected by the decoder.
# 
# VAE Decoding: It passes these scaled latents through the VAE to produce a viewable image.
# 
# Normalization and Processing: It cleans up the resulting image data, moving it to your CPU and formatting it for standard display libraries.
# 
# Storage: Finally, it saves each step to a directory, allowing you to build a visual grid of the entire generation path.
# =============================================================================
def save_intermediate_steps(step_index, timestep, latents):
    """
    Callback function to save intermediate images during the denoising process.

    Args:
        step_index: The current step number in the generation process.
        timestep: The specific timestep value associated with the current step.
        latents: The latent tensor representation at the current step.
    """
    with torch.no_grad():
        # Scale the latents using the VAE scaling factor specific to Stable Diffusion
        # Rescale latents to the range expected by the VAE decoder
        latents_input = latents / 0.18215
        
        # Decode the latent representation into an image using the VAE
        image = pipe.vae.decode(latents_input).sample
        
        # Normalize the image data to the range [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Move image to CPU, rearrange dimensions, and convert to numpy array
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
    # Convert the numpy array to a PIL Image object
    pil_image = Image.fromarray((image * 255).astype("uint8"))
    
    # Define the output directory for saving intermediate steps
    outdir = "intermediate_steps"
    
    # Create the directory if it does not already exist
    os.makedirs(outdir, exist_ok=True)
    
    # Save the current step's image to the output directory
    pil_image.save(f"{outdir}/step_{step_index:02d}.png")
    
    # Append the step index and image to the global list for later grid plotting
    all_steps.append((step_index, pil_image))
    
# %% 
# List to store intermediate images
all_steps = []

generator = torch.Generator(device).manual_seed(42)

prompt = "A puppy dog riding a skateboard in Times Square"
negative_prompt = "cars"

# %%
# Activate the callback by passing it to the pipeline with callback_steps=1. This will generate a comprehensive grid showing every stage of your image's creation.
# Clear previous
all_steps.clear()

# Run the pipe with the callback
pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    generator=generator,
    callback=save_intermediate_steps,
    callback_steps=1  # Every step
)

# Extract images and create titles
imgs = [img for step, img in all_steps]
titles = [f"Step {step+1}" for step, img in all_steps]

helper_utils.display_grid(
    images=imgs,
    titles=titles,
    n_rows=8, 
    n_cols=5,
    figsize=(15, 24),
    main_title='Stable Diffusion 2: Denoising Steps'
)

# %%
# =============================================================================
# Exploring DDPM: The Foundation of Diffusion¶
# Before modern latent models existed, researchers established the groundwork with Denoising Diffusion Probabilistic Models (DDPM). These models operate through a fundamental two phase process that defines how randomness can be harnessed for creation.
# 
# The Forward Diffusion Process
# Imagine take a perfectly clear image and gradually adding tiny amounts of Gaussian noise to it over many steps, typically 1000. Eventually, your original image is completely obscured, leaving only pure random static. This process is based on fixed mathematical rules and is used to create the training data the model will eventually learn from.
# 
# The Reverse Denoising Process
# This is where the learning happens. The model is trained to reverse the forward process, taking one small step backward at a time. It doesn't just "unblur" the image in one go; instead, it learns to predict and remove the noise at each specific step. By starting from pure randomness and iteratively applying this learned reverse process, the model can manifest a high quality image from nothing.
# 
# Key Principles to Remember:
# Iterative Reconstruction: Unlike some generative models that produce an image in a single pass, diffusion models rely on many small, iterative improvements.
# 
# Stability through Probability: This probabilistic approach makes training more stable and predictable than earlier generative designs.
# 
# Learning the Noise: The actual goal of training is for the model to predict the total accumulated noise that was added to a clean image. This provides a more robust "anchor" for the network to learn from compared to trying to predict subtle differences between two noisy steps.
# 
# Pixel Space vs. Latent Space
# Early DDPMs operated directly in pixel space, meaning they had to process every individual pixel of a high resolution image. This is incredibly computationally intensive. Modern models like Stable Diffusion solve this by working in latent space, a compressed mathematical representation that captures the essential visual features while being much smaller and faster to process.
# 
# You'll now load a classic DDPM trained specifically on bedroom images to see these foundational principles in action.
# =============================================================================
# Initializing the DDPM Pipeline
# Identify the specific DDPM model for bedroom generation and load it using your DDPMPipeline helper. You can safely ignore any warnings about .safetensors files; this is standard when working with older model weights.
model_id = "google/ddpm-ema-bedroom-256"  # Bedroom images (256x256)

ddpm_pipeline = load_model_pipeline(
    model_id=model_id,
    pipeline_class=DDPMPipeline,
    device=device
)

print("\nLoading Complete!")

# %% Analyzing the Denoising Dynamics
# Classic DDPMs often require many more steps than Stable Diffusion to reach peak quality. You'll set this to 1000 steps to ensure you see the full fidelity of the reconstruction process.
# Using 1000 steps for highest quality DDPM generation
# (DDPM typically requires more steps than Stable Diffusion for best results)
num_inference_steps = 1000

print(f"Using model: {model_id}")
print(f"Image resolution: 256x256 pixels")
print(f"Steps: {num_inference_steps}")

# Access the model and scheduler
model = ddpm_pipeline.unet
scheduler = ddpm_pipeline.scheduler

# Set up for visualization
scheduler.set_timesteps(num_inference_steps)

# %%
# To truly appreciate how the model works, you'll visualize two distinct viewpoints of the same process:

# 1. Predicted Clean Image: At every step, you'll ask the model to estimate what the final, noise free image will be. Early on, this will be a blurry, average guess, becoming progressively sharper.
# 2. Actual Noisy State: This shows the image exactly as it exists at that specific moment in time—starting as pure static and gradually revealing structural details.
# --- Run pixel-space DDPM comparison ---
num_inference_steps = 100

print("Generating DDPM denoising comparison...")

gradual_images, full_removal_images = helper_utils.visualize_ddpm_denoising(
    ddpm_pipeline, 
    num_inference_steps=num_inference_steps
)

# %%
num_splits = 5

# Use the actual step IDs saved
actual_step_ids = [gi[0] for gi in gradual_images]
num_inference_steps = actual_step_ids[-1]  # last step
step_indices = [int(np.round(i * num_inference_steps / (num_splits - 1))) for i in range(num_splits - 1)]
step_indices.append(num_inference_steps)

print("Plotted step indices:", step_indices)

# Speed up access: build {step: image} dictionaries
grad_dict = {s: img for (s, img) in gradual_images}
full_dict = {s: img for (s, img) in full_removal_images}

# Use closest available step if step is not present
def get_closest_img(dct, step):
    # Find the closest available key in the dict
    best = min(dct.keys(), key=lambda k: abs(k-step))
    return dct[best]

images_full = [get_closest_img(full_dict, s) for s in step_indices]
images_grad = [get_closest_img(grad_dict, s) for s in step_indices]

# --- PLOTTING SECTION ---

# Prepare Titles
# Create a flat list of titles corresponding to (Row 1 images) + (Row 2 images)
titles_row1 = [f"Step {s}:\nPredicted Clean Image" for s in step_indices]
titles_row2 = [f"Step {s}:\nActual Noisy State" for s in step_indices]
all_titles = titles_row1 + titles_row2

# Pass a list of lists [[row1], [row2]] so the function knows to create 2 rows
helper_utils.display_grid(
    images=[images_full, images_grad],
    titles=all_titles,
    row_labels=["Model Prediction", "Actual Noise"],
    main_title='Pixel-Space DDPM: Denoising Analysis',
    figsize=(3.5 * len(step_indices), 7)
)
# %% Interpreting the Denoising Grid
# =============================================================================
# As you examine the two rows of results, you'll see a clear distinction in how the model operates:
# 
# The Jump to Prediction (Top Row): You'll notice that even at the earliest steps, the model is trying to guess the final room. Because it has almost no information, this initial guess is extremely blurry and represents a statistical "average" of all its training images. As the steps progress and it gathers more data from the noisy input, its estimate becomes sharper and more accurate.
# 
# The Gradual Reveal (Bottom Row): This row shows you how the ACTUAL image looks during generation. It remains unrecognizable as a room for much longer than the prediction row, emphasizing that the model isn't just "unblurring" a picture but is statistically reconstructing structural integrity from total randomness over time.
# 
# This reveals the pivotal lesson of diffusion: the goal isn't a one step transformation, but a gradual, iterative journey from chaos to order.
# 
# =============================================================================
# =============================================================================
# Generative Contrast: Pixel Space vs. Latent Space
# Feature               | Pixel-Space (DDPM)                        | 	Latent-Space (Stable Diffusion)
# Data Type             | High-resolution RGB Pixels                |	Compressed Latent Representations
# Computational Cost	| High (expensive to process every pixel)   |   Low (efficiently processes small latents)
# Visual Progress       | Denoising looks like "static" clearing up |	Denoising looks like "blobs" forming shapes
# 
# =============================================================================
