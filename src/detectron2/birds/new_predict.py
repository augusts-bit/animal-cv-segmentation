# Based on https://github.com/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb

# ==============================================================

# Setup and import

# ==============================================================

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale
print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Detectron and torch (check for correct versions)
import torch, detectron2
print("imported torch and detectron2")

# Parse arguments given to job
# import argparse

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, random
import re
import torch
import gc
import shutil

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
from sahi.utils.detectron2 import export_cfg_as_yaml

# SAHI
from sahi.utils.detectron2 import Detectron2TestConstants
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image

# Other
import sys, os, distutils.core
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2
from osgeo import gdal
import geopandas as gpd
from pyproj import CRS
import matplotlib.image as mpimg
import rasterio
from shapely.geometry import Polygon
import random
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from osgeo import gdal

# Install GDAL like this?
import subprocess
import sys

# Import helpers
from helpers.def_custom_sliced_predict import *

# ==============================================================

# Load image / raster and tile

# ==============================================================

# Save folder
if os.path.isdir("slices"): # remove old slices folder so that it always contains slices of current run (will be made during prediction)
    shutil.rmtree("slices")
    os.makedirs("slices")

# Load raster, convert to image and tile
rastername = "Texel_Wagejot_2022_vlucht1_clipped.tif"
rasterpath = os.path.join("input", rastername)
raster_data_set = gdal.Open(rasterpath)

gt = raster_data_set.GetGeoTransform()
pixelSizeX = gt[1] # cell size

# Convert to array image
raster_band_1 = raster_data_set.GetRasterBand(1) # red channel
raster_band_2 = raster_data_set.GetRasterBand(2) # green channel
raster_band_3 = raster_data_set.GetRasterBand(3) # blue channel
raster_b1 = raster_band_1.ReadAsArray()
raster_b2 = raster_band_2.ReadAsArray()
raster_b3 = raster_band_3.ReadAsArray()

# Stack to image
raster_img = np.dstack((raster_b1, raster_b2, raster_b3))

# cell size of Texel PHZ
target_cs = 0.011272670412636411

# Tile pixel size (depending on cell size)
W = round(445 * (target_cs/pixelSizeX)) # ~ 5.0 m x 5.0 m # was 350

# Split into tiles
# raster_tiles = [raster_img[x:x+W,y:y+W] for x in range(0,raster_img.shape[0],W) for y in range(0,raster_img.shape[1],W)]

raster_tiles = []
tile_locations = []

# Loop through all tiles
for x in range(0, raster_img.shape[0], W):
    for y in range(0, raster_img.shape[1], W):
        tile = raster_img[x:x+W, y:y+W]
        raster_tiles.append(tile)
        tile_locations.append((x, y))

# Save
n = 1
for raster_tile in raster_tiles:
    raster_tile_name = "slices/" + os.path.splitext(rastername)[0] + str(n) + ".png"
    raster_tile = cv2.convertScaleAbs(raster_tile) # convert to uint8
    plt.imsave(raster_tile_name, raster_tile)
    n = n + 1

# ==============================================================

# Inference/predict

# ==============================================================

# Check if a GPU is available
if torch.cuda.is_available():
    device_name = 'cuda' # Otherwise will get Runtime error as no NVIDIA
    device_name = 'cpu' # leave on cpu
else:
    device_name = 'cpu'

# Get model
cfg = get_cfg()
cfg.merge_from_file("model/detectron2/model_cfg.yaml")
cfg.MODEL.WEIGHTS =  "model/detectron2/model_weights.pt"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.MODEL.DEVICE = device_name
predictor = DefaultPredictor(cfg)

# Create a binary mask for the entire original image
full_mask = np.zeros_like(raster_img[:,:,0], dtype=np.uint8)

# Loop through all tiles
for i, tile in tqdm(enumerate(raster_tiles), total=len(raster_tiles)):

    # Load tile image
    tile_path = "slices/" + os.path.splitext(rastername)[0] + str(i+1) + ".png"
    im = cv2.imread(tile_path)

    # Apply model to the tile
    outputs = predictor(im)

    # Get model's output
    masks = outputs["instances"].get('pred_masks').cpu().numpy()
    class_ids = outputs["instances"].get('pred_classes').cpu().numpy()

    # Calculate the position of the tile in the original image
    x, y = tile_locations[i]

    # Update full mask if masks were predicted
    if len(masks) != 0:
        # Sum the masks along the first axis before combining with full_mask
        combined_mask = np.sum(masks, axis=0)

        for instance in range(len(masks)):
            # Identify the pixels where the mask is true
            mask_true_indices = masks[instance] == 1

            # Assign class-specific values to the relevant portion of full_mask
            full_mask[x:x + W, y:y + W][mask_true_indices] = class_ids[instance] + 1

            # Update the relevant portion of full_mask
            # full_mask[x:x + W, y:y + W] = np.logical_or(full_mask[x:x + W, y:y + W], masks[instance].astype(np.uint8))

plt.imsave("test.png", full_mask)




    
