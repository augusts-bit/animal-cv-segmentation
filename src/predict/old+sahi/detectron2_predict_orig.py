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

# Parse arguments
import argparse

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

# Arguments

# ==============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=0.3)
args = parser.parse_args()

if args.input is None:
    raster_files = [file for file in os.listdir("input") if file.lower().endswith(('.tif', '.tiff', ".TIF", ".TIFF"))]
    if raster_files:
        args.input = raster_files[0]
    else:
        print("No (valid) input given or in folder. Load raster in input folder.")
        sys.exit()
else:
    if args.input not in os.listdir("input"):
        print("-----------------------------------")
        print(args.input, "does not exist.")
        print("-----------------------------------")
        sys.exit()

print("-----------------------------------")
print("Doing", args.input, "in input folder. To do other raster in input folder use --input '[x].tif'.")
print("-----------------------------------")
# ==============================================================

# Load image / raster and tile

# ==============================================================

print("-----------------------------------")
print("Converting", args.input, "to image and making tiles... (see slices folder)")
print("-----------------------------------")

# Save folder
if os.path.isdir("slices"): # remove old slices folder so that it always contains slices of current run (will be made during prediction)
    shutil.rmtree("slices")
    os.makedirs("slices")

# Load raster, convert to image and tile
rastername = args.input
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

# Overlap percentage
overlap_percentage = args.overlap

raster_tiles = []
tile_locations = []

# Calculate the overlap size in pixels
overlap_size_W = int(W * overlap_percentage)

# Loop through all tiles with overlap
for x in range(0, raster_img.shape[0] - W + 1, W - overlap_size_W):
    for y in range(0, raster_img.shape[1] - W + 1, W - overlap_size_W):
        tile = raster_img[x:x + W, y:y + W]
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold   # set a custom testing threshold
cfg.INPUT.MIN_SIZE_TEST = 538 # 0 is no rescale
cfg.INPUT.MIN_SIZE_TRAIN = (538,) # trained on 538
cfg.MODEL.DEVICE = device_name
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000 # Detect more instances? (https://github.com/facebookresearch/detectron2/issues/1481)
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
predictor = DefaultPredictor(cfg)

print("-----------------------------------")
print("Loaded model, will do inference... See output/[x]_mask.png for progress and output in case of crash.")
print("-----------------------------------")

# Create a binary mask for the entire original image
full_mask = np.zeros_like(raster_img[:,:,0], dtype=np.uint8)

# Skip white images
def all_white_pixels(image):
    '''Returns True if all white pixels or False if not all white'''
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    pixels = cv2.countNonZero(thresh)
    return True if pixels == (H * W) else False

# Delete large variables
del raster_img, raster_b1, raster_b2, raster_b3, raster_band_1, raster_band_2, raster_band_3
gc.collect()

# Loop through all tiles
for i, tile in tqdm(enumerate(raster_tiles), total=len(raster_tiles)):

    # Load tile image
    tile_path = "slices/" + os.path.splitext(rastername)[0] + str(i+1) + ".png"
    im = cv2.imread(tile_path)

    # Skip if white image
    if all_white_pixels(im):
        continue

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

            # Update relevant portion of full_mask
            full_mask[x:x + W, y:y + W][mask_true_indices] = class_ids[instance] + 1 # +1 to leave 0 as background

    # Save mask and progress
    if (i + 1) % 20 == 0:
        plt.imsave("output/"+os.path.splitext(rastername)[0]+"_mask.png", full_mask) # you can use the mask to create shapefiles in case of crash/stop (continue at next step)

plt.imsave("output/"+os.path.splitext(rastername)[0]+"_mask.png", full_mask) # Final mask save

# ==============================================================

# Convert mask to shapefile with classes

# ==============================================================

print("-----------------------------------")
print("Done with inference, converting mask to shapefile...")
print("-----------------------------------")

# Set categories
if os.path.isfile('model/detectron2/model_categories.json'):
    with open('model/detectron2/model_categories.json', 'r') as json_file:
        categories = json.load(json_file) # make sure the categories are from the training and follow the indexing
else:
    # This might be incorrect incase different/new species are added in training
    categories = {'0': 'anders', '1': 'dwergmeeuw', '2': 'grote stern', '3': 'kluut', '4': 'kokmeeuw', '5': 'visdief', '6': 'zwartkopmeeuw'}

# Identify contours for each bird and what species they are
contours = []
majority_values = []

for class_value in np.unique(full_mask):
    if class_value == 0:  # Skip background
        continue

    mask_class = (full_mask == class_value).astype(np.uint8)
    contours_class, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_class:
        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI)
        roi = full_mask[y:y + h, x:x + w]

        # Find the unique values and their counts in the ROI
        unique_values, counts = np.unique(roi, return_counts=True)

        # Exclude 0 from consideration (is background)
        counts = counts[unique_values != 0]
        unique_values = unique_values[unique_values != 0]

        # Get the index of the maximum count (majority value) --> necessary in cases of overlap
        majority_index = np.argmax(counts)
        majority_value = unique_values[majority_index]

        # Append the contour and its majority value
        contours.append(contour)
        majority_values.append(majority_value)

# Create GeoDataFrame to store polygons
polygons = gpd.GeoDataFrame(columns=['soort_id', 'soort', 'grootte', 'geometry'])

# Iterate over contours and create polygons
for class_id, (contour, majority_value) in enumerate(zip(contours, majority_values), start=1):

    # Geometry, loop through the points in the contour
    geographic_points = []  
    for point in contour.squeeze():
        if np.isscalar(point):  # Check if point is a scalar
            continue
        x_pixel, y_pixel = point[0], point[1]
        x_geo = gt[0] + (x_pixel * gt[1]) + (y_pixel * gt[2]) # gt is geotransform of input raster (defined earlier)
        y_geo = gt[3] + (x_pixel * gt[4]) + (y_pixel * gt[5])
        geographic_points.append([x_geo, y_geo])

    # Check if there are enough points to create a polygon
    if len(geographic_points) < 4:
        continue
            
    polygon = Polygon(geographic_points) # Create a polygon from the converted points

    # Soort
    species_name = categories.get(str(majority_value-1), 'Unknown') # 0 is background in mask, so values are +1 of actual IDs

    # Append
    instance_row = pd.DataFrame({'soort_id': majority_value, 'soort': species_name, 'grootte': polygon.area, 'geometry': polygon}, index=[0])
    polygons = pd.concat([polygons, instance_row], ignore_index=True)

# Transform geometry
with rasterio.open(os.path.join("input", rastername)) as src:
    transform = src.transform
    crs = src.crs

# Transform the geometry of the polygons to match that of the original raster (will still need to do 'Define Projection' in ArcGIS Pro)
polygons = polygons.set_crs(crs)

# Save the transformed GeoDataFrame
polygons.to_file("output/"+os.path.splitext(rastername)[0]+"_polygons.shp", crs=crs)

print("-----------------------------------")
print("Done! See output folder.")
print("-----------------------------------")
