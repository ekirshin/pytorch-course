#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 22:24:17 2026

@author: ek
"""

from pprint import pprint

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import Image as DisplayImage
from PIL import Image
from torchvision.io import decode_image

import helper_utils

#%%
# Load the image
image = decode_image('./images/dog1.jpg')

# Sample bounding boxes, where each box is in (xmin, ymin, xmax, ymax) format
# The first box is set to draw around the whole dog, the other to draw around the left eye
boxes = torch.tensor([[140, 30, 375, 315], [200, 70, 230, 110]], dtype=torch.float)

# Corresponding labels for the detected objects
labels = ["dog", "eye"]

# Draw boxes on the image
result = vutils.draw_bounding_boxes(image=image, 
                                    boxes=boxes, 
                                    labels=labels,           # This is optional
                                    colors=["red", "blue"],  # This is optional. By default, random colors are generated for boxes.
                                    width=3                  # This is optional. The default is width=1
                                   )

# Display the result
helper_utils.display_images(processed_image=result, figsize=(10, 10))

#%% 
# Load the pre-saved segmentation mask
mask_filename = "dog_segmentation_mask.pt"
loaded_object_mask = torch.load(mask_filename)

# Make it (1, H, W)
object_mask = loaded_object_mask.unsqueeze(0)


# Draw segmentation mask on the image
result  = vutils.draw_segmentation_masks(image=image,
                                         masks=object_mask,
                                         alpha=0.5,          # This is optional. The default is alpha=0.8
                                         colors=["blue"]     # This is optional. By default, random colors are generated for each mask.
                                        )

# Display the result
helper_utils.display_images(processed_image=result, figsize=(10, 10))

#%% Get information about the model predicted classes

def get_model_classes_from_weights_meta(model, weights_obj=None):
    """
    Inspects a model's weights object to find and return the class names.
    
    Args:
        model: The model instance.
        weights_obj: The weights enumeration object (e.g., ResNet50_Weights.DEFAULT).
    
    Returns:
        A tuple containing (number_of_classes, list_of_class_names), or (None, None) if not found.
    """
    num_classes = None
    class_names = None

    # Check if a weights object was provided and if it has the necessary metadata
    if weights_obj and hasattr(weights_obj, 'meta') and "categories" in weights_obj.meta:
        class_names = weights_obj.meta["categories"]
        num_classes = len(class_names)
        print(f"Model is configured for {num_classes} classes based on Weights Metadata. These classes are:\n")
        # For nice printing, let's display the list
        pprint(class_names)
            
        return num_classes, class_names

    else:
        print("'categories' metadata not found for this model.")
        return num_classes, class_names
    
#%%  # Defining the model itself without weights for now
seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=None)

# Select the specific pre-trained weights you want to inspect for your selected model
seg_model_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT

# Call the helper function to inspect the provided weights object.
num_classes, class_names_deeplabv3 = get_model_classes_from_weights_meta(
    model=seg_model,              # The model architecture.
    weights_obj=seg_model_weights # The weights object containing the .meta attribute to inspect.
)

#%% Dealing with models without the meta data incorporated into weights
# Instantiate the ResNet50 model using the legacy `pretrained=True` method.
resnet50_model = tv_models.resnet50(pretrained=True)
# When using pretrained=True, the weights are loaded directly into the 
# model's layers. Unlike the modern approach, this doesn't create 
# a separate weights object that holds the metadata. 
# Therefore, you'll explicitly pass weights_obj=None to the helper
# function to demonstrate that no such object is available with this 
# method.

# Pass `weights_obj=None` because the `pretrained=True` loading method
# does not create a separate weights object that has a .meta attribute to inspect.
num_classes, class_names = get_model_classes_from_weights_meta(
    model=resnet50_model, 
    weights_obj=None
)

#%% Manually Inspecting the Architecture
print(resnet50_model)

#%% Get the number of output features from the layer named 'fc'
num_classes = resnet50_model.fc.out_features

print(f"Inspecting the model's .fc layer: It has {num_classes} output classes.")

#%% Using trained off-the-shelf models

#%% Image Segmentation
# Perform image segmentation with the DeepLabV3 model. The goal is straightforward: find a dog in an image and generate a pixel-perfect mask that highlights its exact shape.
# Define the file path for the image.
image_path = './images/dog2.jpg'

# Display the image.
DisplayImage(image_path, width=300, height=450)

#%% Instantiate the model architecture and load the pre-trained weights.
seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=seg_model_weights).eval()

#%% # Load the base PIL Image
img = Image.open(image_path)

# Create the clean, un-normalized tensor for visualization later
original_image_tensor = transforms.ToTensor()(img)

# Define the normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

# Create the normalized input tensor for the model
# .unsqueeze(0) adds a batch dimension, changing the tensor shape 
# from [C, H, W] to [N, C, H, W], as models expect a batch of images for input
input_tensor = normalize(original_image_tensor).unsqueeze(0)

# Define a list of target classes you want to find
target_class_names = ['dog'] # More classes can be added as well, e.g., ['dog', 'person', ...]

# Define a corresponding list of colors for the segmentation masks for each class
seg_colors = ["blue"]

# Use a list comprehension to get a list of corresponding class indices
class_indices = [class_names_deeplabv3.index(name) for name in target_class_names]

# Print the results for confirmation
print(f"Target Classes:        {target_class_names}")
print(f"Corresponding Indices: {class_indices}")

#%% Perform Inference: 
# Pass the normalized input_tensor to the pre-trained seg_model. 
# The model returns its predictions in a dictionary, so you'll 
# use ['out'][0] to access the raw output scores (logits) for 
# the first (and only) image in the batch.
# Generate prediction
with torch.no_grad():
    output = seg_model(input_tensor)['out'][0]
    
#%% Generate the Final Mask
# Get the predicted class for each pixel by finding the class with the highest score.
output_predictions = output.argmax(0)

# Create a separate boolean mask for each of your target classes.
# The result is a list of boolean tensors, one for each class index.
individual_masks = [(output_predictions == i) for i in class_indices]

# Stack the individual masks into a single tensor of shape (num_masks, H, W).
stacked_masks = torch.stack(individual_masks, dim=0)

#%% Draw Mask
# Apply segmentation masks using the stacked_masks tensor.
result = vutils.draw_segmentation_masks(image=(original_image_tensor * 255).byte(),
                                        masks=stacked_masks,
                                        alpha=0.5,
                                        colors=seg_colors)
# Visualize the mask
helper_utils.display_images(processed_image=result, figsize=(7, 7))

#%% Image Classification
# Load the ResNet50 model, using the legacy weights and set it to .eval()
resnet50_model = tv_models.resnet50(pretrained=True).eval()

# Use the helper function to load the class index-to-name mappings from the JSON file.
imagenet_classes = helper_utils.load_imagenet_classes('./imagenet_class_index.json')

print("Total Classes:", len(imagenet_classes), "\n")

# Define the starting index and how many classes you want to print
start_index = 200
num_to_print = 10

print(f"Printing {num_to_print} classes starting from index {start_index}:\n")

# Loop through the desired range of indices
for i in range(start_index, start_index + num_to_print):
    key = str(i)
    value = imagenet_classes[key]
    print(f"Index {key}: {value}")
    
#%% # Load the base PIL Image
img = Image.open(image_path)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    # Models like ResNet50 expect a 224x224 input, so you resize to a slightly
    # larger image and then take a center crop of the target dimensions.
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Normalize the tensor with the mean and standard deviation from the ImageNet dataset.
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply the transforms and add a batch dimension
input_tensor = transform(img)
# Model's input layer expects a 4D tensor with a specific shape: [N, C, H, W]
input_batch = input_tensor.unsqueeze(0)

#%%  Perform Inference
with torch.no_grad():
    output = resnet50_model(input_batch)
        
#%% # Apply softmax for probabilities
# Apply it to the first (and only) item in the output batch.
probabilities = torch.nn.functional.softmax(output[0], dim=0) 

#%% # Get the `top` probabilities and their corresponding class IDs
top = 5
top_prob, top_catid = torch.topk(probabilities, top)

#%% # Convert IDs to class names and print results
print(f"Top {top} predictions:")
for i in range(top_prob.size(0)):
    # Get the string representation of the class ID
    class_id_str = str(top_catid[i].item())
    
    # Look up the class name in the dictionary
    class_name = imagenet_classes[class_id_str][1]
    confidence = top_prob[i].item() * 100
    print(f"\tTop-{i+1}: {class_name} ({confidence:.2f}%)")

#%% Detecting Objects using fasterrcnn_resnet50_fpn Model
# Load a pre-trained object detection model and set to evaluation mode
bb_model_weights = tv_models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
bb_model = tv_models.detection.fasterrcnn_resnet50_fpn(weights=bb_model_weights).eval()

# Use the helper function to inspect the weights object of the object detection model.
num_classes, classes = get_model_classes_from_weights_meta(
    model=bb_model, 
    weights_obj=bb_model_weights
)

#%% 
# Set the file path to the image you'll use for car/traffic light detection.
image_path = './images/cars.jpg'

# Display the image
DisplayImage(image_path, width=600, height=550)

#%% # Define a list of target classes to detect
target_class_names = ['car', 'traffic light']

# Define a corresponding list of colors for each class's bounding box
bbox_colors = ['red', 'blue']

# Use a list comprehension to get a list of all target indices
object_indices = [classes.index(name) for name in target_class_names]

#%% The function collects all qualifying boxes from all your target classes and then uses the draw_bounding_boxes utility to draw them onto the image, each with its specified label and color. The final modified image (or original, if no boxes qualified) is returned.

def detect_and_draw_bboxes(model, image_path, object_indices, labels, bbox_colors, threshold, bbox_width=3):
    """
    Detects and draws labeled bounding boxes for multiple specified object classes on an image.

    Args:
        model: Pre-trained object detection model.
        image_path (str): Path to the image file.
        object_indices (list): List of indices for the target classes to detect.
        labels (list): List of text labels for each target class.
        bbox_colors (list): List of colors for each target class's bounding boxes.
        threshold (float): Confidence threshold for detections.
        bbox_width (int, optional): The line width for the bounding boxes. Defaults to 3.

    Returns:
        torch.Tensor: Image tensor with all detected boxes drawn.
    """
    
    # Open and transform the image, and prepare the result tensor
    pil_image = Image.open(image_path).convert("RGB")
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor_image_batch = transform_to_tensor(pil_image).unsqueeze(0)
    result_image_tensor = (tensor_image_batch.squeeze(0) * 255).byte()

    # Perform inference to get predictions for all possible objects
    with torch.no_grad():
        prediction = model(tensor_image_batch)[0]

    # Initialize lists to collect all boxes, labels, and colors that meet the criteria
    all_boxes_to_draw = []
    all_labels_to_draw = []
    all_colors_to_draw = []

    # Loop through each target class to find its boxes
    for index, label, color in zip(object_indices, labels, bbox_colors):
        # Filter predictions for the current class index and confidence threshold
        class_mask = (prediction['labels'] == index) & (prediction['scores'] > threshold)
        
        # Get the boxes for the current class
        boxes_for_this_class = prediction['boxes'][class_mask]

        if boxes_for_this_class.nelement() > 0:
            # Add the found boxes to our master list
            all_boxes_to_draw.extend(boxes_for_this_class.tolist())
            # Create and add corresponding labels and colors
            all_labels_to_draw.extend([label] * len(boxes_for_this_class))
            all_colors_to_draw.extend([color] * len(boxes_for_this_class))

    # After checking all classes, draw all collected boxes at once if any were found
    if all_boxes_to_draw:
        result_image_tensor = vutils.draw_bounding_boxes(
            result_image_tensor,
            torch.tensor(all_boxes_to_draw),
            labels=all_labels_to_draw,
            colors=all_colors_to_draw,
            width=bbox_width
        )
    else:
        # If the list of boxes to draw is empty, print this information.
        print(f"No objects from the list {labels} were found with a confidence score above {threshold}.\n")
    
    return result_image_tensor

#%%
confidence_threshold = 0.7

#%% # Execute the main detection function
result_image_tensor = detect_and_draw_bboxes(
    model=bb_model,                  # The pre-trained object detection model.
    image_path=image_path,           # The path to the input image.
    object_indices=object_indices,   # The list of integer indices for the target classes.
    labels=target_class_names,       # The list of string names for the box labels.
    bbox_colors=bbox_colors,         # The list of colors for the bounding boxes.
    threshold=confidence_threshold,  # The minimum confidence score for a detection.
)

# Display the results
helper_utils.display_images(processed_image=result_image_tensor, figsize=(15, 15))

#%% 
image_path = './images/car_bus_tram.jpg' ### Add your image path here

# Display the image
DisplayImage(image_path, width=500, height=500)

#%% # Define a list of target classes to detect
target_class_names = ['car', 'bus', 'train'] ### Add your target class names here

# Define a corresponding list of colors for each class's bounding box
bbox_colors = ['red', 'purple', 'green'] ### Add your target class names here

# Set the confidence_threshold
confidence_threshold = 0.7 ### Set your threshold here

#%% 
detection_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#%% # Use a list comprehension to get a list of all target indices
object_indices = [detection_classes.index(name) for name in target_class_names]

# Define the label for the object same as the target_class_names
labels = target_class_names

# Execute the main detection function
result_image_tensor = detect_and_draw_bboxes(
    model=bb_model,                  # The pre-trained object detection model.
    image_path=image_path,           # The path to the input image.
    object_indices=object_indices,   # The list of integer indices for the target classes.
    labels=labels,                   # The list of string names for the box labels.
    bbox_colors=bbox_colors,         # The list of colors for the bounding boxes.
    threshold=confidence_threshold,  # The minimum confidence score for a detection.
    bbox_width=5                     # The line thickness for the bounding boxes.
)

# Display the results (feel free to set a different `figsize`)
helper_utils.display_images(processed_image=result_image_tensor, figsize=(10, 10))

#%% Generating Segmentation on Objects on your Own Images

image_path = './images/person_and_bicycle.jpg' ### Add your image path here

# Display the image
DisplayImage(image_path, width=500, height=500)

#%% 
segmentation_classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Define a list of target classes you want to find
target_class_names = ['person', 'bicycle'] ### Add your target class name here

# Define a corresponding list of colors for the segmentation masks for each class
seg_colors = ["pink", 'yellow'] ### Add your segmentation masks for colors each class here

#%% # Load pre-trained segmentation model
seg_model_weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
seg_model = tv_models.segmentation.deeplabv3_resnet50(weights=seg_model_weights).eval()

# Load the base PIL Image
img = Image.open(image_path)

# Create the clean, un-normalized tensor for visualization later
original_image_tensor = transforms.ToTensor()(img)

# Define the normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

# Create the normalized input tensor for the model
input_tensor = normalize(original_image_tensor).unsqueeze(0)

# Use a list comprehension to get a list of corresponding class indices
class_indices = [segmentation_classes.index(name) for name in target_class_names]

#%% # Generate prediction
with torch.no_grad():
    output = seg_model(input_tensor)['out'][0]

#%% # Get the predicted class for each pixel by finding the class with the highest score.
output_predictions = output.argmax(0)

# Create a separate boolean mask for each of your target classes.
individual_masks = [(output_predictions == i) for i in class_indices]

# Stack the individual masks into a single tensor of shape (num_masks, H, W).
stacked_masks = torch.stack(individual_masks, dim=0)

# Apply segmentation masks using the stacked_masks tensor.
result = vutils.draw_segmentation_masks(image=(original_image_tensor * 255).byte(),
                                        masks=stacked_masks,
                                        alpha=0.5,
                                        colors=seg_colors)

# Visualize the mask  (feel free to set a different `figsize`)
helper_utils.display_images(processed_image=result, figsize=(10, 10))

