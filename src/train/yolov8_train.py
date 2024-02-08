
"""YOLOv8_SAHI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xuecuiu4n3ObxuL8cLmHhmfAKKIX2nui
"""

# Based on https://github.com/computervisioneng/image-segmentation-yolov8/

# ------------------------------------------------------------------------------------- #
# Import
# ------------------------------------------------------------------------------------- #

import os, re, random
import pandas as pd
import cv2
from ultralytics import YOLO
# from utils.datasets import *
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import *
import argparse # Parse arguments given to job

# Get helpers functions
from helpers.getCategoriesSizes import *

# ==============================================================

# Load inputs given to job

# ==============================================================

# This is how you load input from job
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-train-model?view=azureml-api-2 

# input and output arguments
parser = argparse.ArgumentParser()

# Training data
parser.add_argument("--traindata")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--lr0", type=float, default=0.01)
parser.add_argument("--imgsz", type=int, default=538)
args = parser.parse_args()

print(os.listdir(args.traindata))

# ==============================================================

# Check what species are in dataset

# ==============================================================

# ann_files = sorted(os.listdir(os.path.join(args.traindata, "annotations")))
categories, min_width, min_height = get_classes_sizes(os.path.join(args.traindata, "annotations"))
soorten = sorted(list(set(categories)))
print("Dataset contains:", soorten)
print("Minimum width and heigth:", min_width, "x", min_height) # maybe base rescaling of this

# save dictionary of classes
soorten_dict = {str(index): bird for index, bird in enumerate(soorten)}
with open('outputs/model_categories.json', 'w') as json_file:
    json.dump(soorten_dict, json_file)

# ------------------------------------------------------------------------------------- #
# Split train-validation and create yaml
# ------------------------------------------------------------------------------------- #

# Rename images and labels to folder to lowercase (necessary?)
# os.rename(os.path.join(args.traindata, "Images"), os.path.join(args.traindata, "images"))
# os.rename(os.path.join(args.traindata, "Labels"), os.path.join(args.traindata, "labels"))

# See https://github.com/ultralytics/yolov5/issues/1579

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def autosplit(path, weights, annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory 
                        --> assumes that the corresponding labels is in the same parent directory /labels/ 
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    outloc = "outputs" # change so that it doesn't have to write to BLOB (no permission)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', ]  # 2 txt files

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in zip(indices, files):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(os.path.join(outloc, txt[i]), 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file
                
    txt = ['autosplit_train.txt', 'autosplit_val.txt', ]  # 2 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in zip(indices, files):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file

# autosplit(os.path.join(args.traindata, "images"), weights=(0.9, 0.1, 0.0)) # no write access...

data = {
    'path': args.traindata, # root path, start at home directory
    'train': "autosplit_train.txt", # no write access, so store manually
    'val': "autosplit_val.txt",
    'nc': len(soorten), # number of classes
    'names': soorten
}

with open(os.path.join("outputs", "config.yaml"), 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

# ------------------------------------------------------------------------------------- #
# Train YOLOv8
# ------------------------------------------------------------------------------------- #

# Check if a GPU is available
if torch.cuda.is_available():
    device_name = 'cuda' 
else:
    device_name = 'cpu'

# load a pretrained model (recommended for training)
model = YOLO('yolov8n-seg.pt')

# arguments for training (see https://docs.ultralytics.com/modes/train/#arguments)
args = {"data": os.path.join("outputs", "config.yaml"),
        "epochs": args.epochs, 
        "batch": args.batch, 
        "lr0": args.lr0,
        "imgsz": args.imgsz,
        "device": device_name,
        "project": "./outputs", "exist_ok": True} # model will be saved in './outputs' folder in Azure

# train
model.train(**args)

# model weights: '/outputs/weights/best.pt'
# model yaml: '/outputs/args.yaml'

# ------------------------------------------------------------------------------------- #
# Log best loss
# ------------------------------------------------------------------------------------- #

# Path to the 'results.csv' file
results_csv_path = os.path.join('outputs', 'train', 'results.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(results_csv_path)

# Remove leading spaces from column names
df.columns = df.columns.str.strip()

# Specify the loss components you want to consider (YOLOv8 bases best model of a weighted loss, but cannot find it)
loss_components = ['val/box_loss', 'val/seg_loss', 'val/cls_loss', 'val/dfl_loss']

# Create a new column that contains the sum of the specified loss components
df['total_loss'] = df[loss_components].sum(axis=1)

# Find the epoch with the minimum total loss
min_loss_epoch = df[df['total_loss'] == df['total_loss'].min()]['epoch'].values[0]

# Print the minimum total loss and the corresponding epoch
print(f"Minimum Total Loss: {df['total_loss'].min()} at Epoch {min_loss_epoch}")

# Log
mlflow.log_metric("total val loss", total_val_loss)