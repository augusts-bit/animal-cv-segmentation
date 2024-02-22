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

def make_labels(annotation_dir, mask_dir, output_dir, class_labels):
    for annotation_file in sorted(os.listdir(annotation_dir)):
        if annotation_file.endswith('.txt'):
            annotation_path = os.path.join(annotation_dir, annotation_file)
            image_path = annotation_path.replace("_annotation", "_mask").replace(".txt", ".png")

            # Read class labels and bounding boxes from annotation file
            with open(annotation_path, 'r') as ann_file:
                ann_content = ann_file.read()

                # Use regex to find the number of objects
                match = re.search(r'Objects with ground truth : (\d+)', ann_content)
                if match:
                    num_objects = int(match.group(1))
                else:
                    num_objects = 0

                # Initialize lists to store polygons and class indices for all objects
                all_polygons = []
                all_class_indices = []

                # Process each object in the annotation file
                for i in range(num_objects):
                    original_label_match = re.search(r'Original label for object {} : "(.*?)"'.format(i + 1), ann_content)
                    bbox_coords_match = re.search(r'Bounding box for object {} : \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)'.format(i + 1), ann_content)
                    # mask_path_match = re.search(r'Pixel mask for object {} : "(.*?)"'.format(i + 1), ann_content)

                    if original_label_match and bbox_coords_match: # and mask_path_match:
                        original_label = original_label_match.group(1)
                        bbox_coords = map(int, bbox_coords_match.groups())
                        mask_name = annotation_file.replace("annotation", "mask")
                        mask_path = os.path.join(mask_dir, mask_name[:-4]+".png")

                        class_index = class_labels.index(original_label)

                        # Load the binary mask and get its contours
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                        H, W = mask.shape
                        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Convert the contours to polygons
                        all_polygons_per_object = []
                        cnt = contours[i] # only use the contour of the current object
                        if cv2.contourArea(cnt) > 0:
                            polygon = []
                            for point in cnt:
                                x, y = point[0]
                                polygon.append(x / W)
                                polygon.append(y / H)
                            all_polygons_per_object.append(polygon)

                        # Append polygons and class index to the lists
                        all_polygons.extend(all_polygons_per_object)
                        all_class_indices.extend([class_index] * len(all_polygons_per_object))

                # Print all polygons with corresponding class indices to a single YOLO label file
                img_name = annotation_file.replace("annotation", "img") # label needs to have the same name as the img
                with open(os.path.join(output_dir, '{}.txt'.format(img_name[:-4])), 'w') as f:
                    for class_index, polygon in zip(all_class_indices, all_polygons):
                        line = '{} {}'.format(class_index, ' '.join(map(str, polygon)))
                        f.write('{}\n'.format(line))