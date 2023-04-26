"""
# zero-shot text-conditioned object detection model using OWL-ViT (open-vocabulary object detection network)
    # utilizes multiple text-based queries/prompts to search for and detect target objects in one or multiple images.
    # The goal is to detect novel classes defined by an unbounded (open) vocabulary at inference.
    # uses CLIP with a ViT-like Transformer as its backbone to get multi-modal visual and text features.
"""

# Set-up environment
# !pip install -q git+https://github.com/huggingface/transformers.git

# optional: Install Pillow, matplotlib and OpenCV if you are running this script locally."""
# !pip install Pillow
# !pip install matplotlib
# !pip install opencv-python

import os
import cv2
import skimage
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers import OwlViTProcessor, OwlViTForObjectDetection

#%% helper function (s)

def plot_predictions(input_image, text_queries, scores, boxes, labels):
    
    """
    draws boundary boxes, text queries, and predicted probabiities on top of the input image(s)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
        
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
      
#%% settings

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mixin = ImageFeatureExtractionMixin()

#%% Load pre-trained model (OwlViTForObjectDetection)

# model outputs:
    # - the prediction logits, 
    # - boundary boxes and class embeddings, 
    # - along with the image and text embeddings outputted by the `OwlViTModel`, which is the CLIP backbone.

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Set model in evaluation mode
model = model.to(device)
model.eval()

#%% load image processor

# The processor will resize the image(s), scale it between [0-1] range and normalize it 
    # across the channels using the mean and standard deviation specified in the original codebase.

# Text queries are tokenized using a CLIP tokenizer and stacked to output tensors of the following shape:
    # [batch_size * num_max_text_queries, sequence_length]. 
# If you are inputting more than one set of (image, text prompt/s), 
    # num_max_text_queries -> maximum number of text queries per image across the batch. 
    # Input samples with fewer text queries are padded. 
    
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

#%% pass in multiple sets of images and text queries (corresponding to the same or different objects) to search for in different images. 

    # input text queries as a nested list to `OwlViTProcessor` 
    # and images as lists of (PIL images or PyTorch tensors or NumPy arrays).

# Preprocessing
images = [skimage.data.coffee(), skimage.data.astronaut()]
images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in images]

# Nested list of text queries
# text_queries = [["coffee mug", "spoon", "plate"], ["human face", "rocket", "nasa badge", "star-spangled banner"]]
text_queries = [["coffee mug", "spoon", "plate"], ["human face", "rocket"]]

# Process image and text inputs
inputs = processor(text=text_queries, images=images, return_tensors="pt").to(device)

# Print input names and shapes
# Notice the size of the `input_ids `and `attention_mask` is `[batch_size * num_max_text_queries, max_length]`. 
    # Max_length is set to 16 for all OWL-ViT models.
for key, val in inputs.items():
    print(f"{key}: {val.shape}")

#%% Get prediction

with torch.no_grad():
  outputs = model(**inputs)

for k, val in outputs.items():
    if k not in {"text_model_output", "vision_model_output"}:
        print(f"{k}: shape of {val.shape}")
        
print("\nText model outputs")
for k, val in outputs.text_model_output.items():
    print(f"{k}: shape of {val.shape}")

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}")

#%% plot the predictions for all images

for image_idx in range(len(images)):
    
    image_size = model.config.vision_config.image_size
    image = mixin.resize(images[image_idx], image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    score_threshold = 0.1

    # Get prediction logits
    logits = torch.max(outputs["logits"][image_idx], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][image_idx].cpu().detach().numpy()

    plot_predictions(input_image, text_queries[image_idx], scores, boxes, labels)

