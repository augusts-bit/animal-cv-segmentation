# %%
import sys, os, distutils.core

# Parse arguments given to job
import argparse

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import re
import torch
import tensorboard
import tabulate

def get_classes_sizes(ann_dir):

  categories = []
  min_width = float('inf')
  min_height = float('inf')

  ann_files = sorted(os.listdir(ann_dir))

  for ann_file in ann_files:

        # Get annotations: bounding box, segmentation (mask) and label
          with open(os.path.join(ann_dir, ann_file), "r") as file:
              annotations = file.readlines()

              # Get category
              for annotation in annotations:
                  # Process each line in the file
                  if annotation.startswith('Original label'):

                      # Extract category
                      match = re.search(r'Original label for object \d+ : "(.*?)"', annotation)
                      if match:
                          label = match.group(1)
                          categories.append(label)

              # Join lines into a single string
              annotations_str = ''.join(annotations)

              # Use regular expressions to extract width and height
              width_match = re.search(r'Image size \(X x Y\) : (\d+) x (\d+)', annotations_str)
              if width_match:
                  width = int(width_match.group(1))
                  height = int(width_match.group(2))

                  # Update the minimum width and height if necessary
                  min_width = min(min_width, width)
                  min_height = min(min_height, height)

  # Return only unique categories
  categories = list(set(categories))

  # Return categories and sizes
  return sorted(categories), min_width, min_height