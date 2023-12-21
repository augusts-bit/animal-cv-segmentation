# %%
import sys, os, distutils.core

# Detectron and torch (check for correct versions)
import torch, detectron2

# Parse arguments given to job
import argparse

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import re
import torch
import tensorboard
import tabulate

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.transforms import RandomFlip
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.data import transforms as T
from sahi.utils.detectron2 import export_cfg_as_yaml

# Other
from PIL import Image
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2
import time
import datetime
import random

# Log metrics
import mlflow
import mlflow.pytorch

# See what needs to be in a dataset:
# https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# Custom function if not already in COCO format (format to own)
def get_custom_dicts(img_dir, ann_dir, mask_dir, soorten):
    
    # First, make sure the image, annotation and mask files match
    image_files = []
    annotation_files = []
    mask_files = []
                     
    images = sorted(os.listdir(img_dir))
    
    # Iterate over each image file
    for image in images:
        # Extract the common name
        image_name = image[:-4] # without extension (.png)
        
        # The corresponding mask and annotation files should be named:
        mask_name = image_name.replace('img', 'mask')
        annotation_name = image_name.replace('img', 'annotation')

        # Check if corresponding mask and annotation files exist
        mask_filename = f"{mask_name}.png"
        annotation_filename = f"{annotation_name}.txt"

        mask_path = os.path.join(mask_dir, mask_filename)
        annotation_path = os.path.join(ann_dir, annotation_filename)

        if os.path.exists(os.path.join(mask_dir, mask_filename)) and os.path.exists(os.path.join(ann_dir, annotation_filename)):
            # Both mask and annotation files exist, add all to lists
            image_files.append(image)
            mask_files.append(mask_filename)
            annotation_files.append(annotation_filename)
                     
    # Number of files in dataset
    print(
        len(image_files), "images", 
        len(mask_files), "masks",  
        len(annotation_files), "annotations",
        "after dataset creation"
    )     

    dataset_dicts = []

    for idx, ann_file in enumerate(sorted(annotation_files)):

        record = {}

        # Get image path
        imgs = list(sorted(image_files))
        img_path = os.path.join(img_dir, imgs[idx])

        # Load image
        im = Image.open(img_path)
        width, height = im.size

        # Get segmentation/masks
        masks = list(sorted(mask_files))
        mask_path = os.path.join(mask_dir, masks[idx])

        mask = read_image(mask_path)
        mask = mask[0].cpu().numpy()
        binary_mask = (mask > 0) # important

        rle_dict = coco_mask.encode(np.asarray(binary_mask, order="F"))

        # Add basic image information to record
        record["file_name"] = img_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Save annotations in a list (remains empty if no annotations - only background)
        objs = []

        # Get annotations: bounding box, segmentation (mask) and label
        with open(os.path.join(ann_dir, ann_file), "r") as file:
            annotations = file.readlines()

            bboxes = []
            categories = []

            # Get bounding box
            for annotation in annotations:
                # Process each line in the file
                if annotation.startswith('Bounding box'):

                    # Extract bounding box coordinates
                    match = re.search(r'\((\d+), (\d+)\) - \((\d+), (\d+)\)', annotation)
                    if match:
                        x, y, x_max, y_max = map(int, match.groups())
                        bbox = [x, y, x_max, y_max]
                        bboxes.append(bbox)

            # Get category
            for annotation in annotations:
                # Process each line in the file
                if annotation.startswith('Original label'):

                    # Extract category
                    match = re.search(r'Original label for object \d+ : "(.*?)"', annotation)
                    if match:
                        label = match.group(1)
                        categories.append(label)

            if bbox: # (there are objects in the image)

                for j in range(len(bboxes)):
                    x, y, x_max, y_max = bboxes[j]
                    w = x_max - x
                    h = y_max - y

                    # Assign category from 'soorten' list
                    for z, soort in enumerate(soorten):
                        if categories[j] == soort:
                            category = z

                    # Create the object dictionary
                    obj = {
                        "bbox": [x, y, x + w, y + h],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": rle_dict,
                        "category_id": category, # Between 0, num_classes - 1
                    }

                    # Save annotation to annotations list
                    objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts