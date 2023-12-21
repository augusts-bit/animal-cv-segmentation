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

def get_classes_sizes(ann_files):

  categories = []
  min_width = float('inf')
  min_height = float('inf')

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

  # Return categories and sizes
  return categories, min_width, min_height