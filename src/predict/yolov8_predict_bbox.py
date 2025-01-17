# ==============================================================

# Setup and import

# ==============================================================

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# YOLO and torch (check for correct versions)
import torch
from ultralytics import YOLO

# Parse arguments
import argparse

# import some common libraries
import numpy as np
import pandas as pd
import os, json, random
import re
import torch
import gc
import shutil
import time
import math

# Other
import sys, os, distutils.core
from torchvision.io import read_image
import cv2
from osgeo import gdal
import geopandas as gpd
from pyproj import CRS
import matplotlib.image as mpimg
import rasterio
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.affinity import scale
import random
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from osgeo import gdal
from skimage.segmentation import clear_border

# Install GDAL like this?
import subprocess
import sys

# Import helpers
from helpers.mask2shape import *

# ==============================================================

# Functions

# ==============================================================

# Skip white/black images
def all_white_pixels(image):
    '''Returns True if all white pixels or False if not all white'''
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    pixels = cv2.countNonZero(thresh)
    return True if pixels == (H * W) else False

def all_white_or_black_pixels(image):
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check for all white pixels
    all_white = np.all(gray == 255)

    # Check for all black pixels
    all_black = np.all(gray == 0)

    return all_white or all_black

# ==============================================================

# Start

# ==============================================================

def main(args):

    # First create output directory
    if os.path.isdir(os.path.join(args.output,'output')) == False:
        os.makedirs(os.path.join(args.output,'output'))

    print("-----------------------------------")
    print("Alle output bevindt zicht in "+str(os.path.join(args.output,'output')))
    print("-----------------------------------")

# ==============================================================

# Load image / raster and tile

# ==============================================================

    print("-----------------------------------")
    print(args.input, "wordt geknipt...")
    print("-----------------------------------")

    # Save folder
    if os.path.isdir(os.path.join(args.output,'output', 'slices')): # remove old slices folder so that it always contains slices of current run (will be made during prediction)
        shutil.rmtree(os.path.join(args.output,'output', 'slices'))
    os.makedirs(os.path.join(args.output,'output', 'slices'))

    # Load raster, convert to image and tile
    rastername = os.path.basename(args.input)
    raster_data_set = gdal.Open(args.input)

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

    # Tile pixel size (depending on cell size)
    W = round(int(args.grootte/pixelSizeX)/32) * 32 # make multiple of 32

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

            if all_white_or_black_pixels(tile): # skip black/white tiles
                continue

            raster_tiles.append(tile)
            tile_locations.append((x, y))

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
    model = YOLO(os.path.join(str(args.model),'best.pt'))

    print("-----------------------------------")
    print("Model is geladen, begint met segmentatie en classificatie...", "\n"
                                                                          "Zie " + str(
        os.path.splitext(rastername)[0]) + "_YOLOv8_mask.png voor progressie", "\n")

    if args.backup == "Ja":
        print("Gebruik " + str(
            os.path.splitext(rastername)[0]) + "_YOLOv8_mask.npy voor back-up (met predict_from_mask.py)")
    print("-----------------------------------")

    # Create masks to keep track of predictions for the entire original image
    score_mask = np.zeros_like(raster_img[:, :, 0], dtype=np.uint8)
    unique_mask = np.zeros_like(raster_img[:, :, 0], dtype=np.float32) # float, store category as the number with a unique decimal

    # Delete large variables
    del raster_img, raster_b1, raster_b2, raster_b3, raster_band_1, raster_band_2, raster_band_3
    gc.collect()

    # Loop through all tiles
    start = time.time() # Check how long it takes
    unique_n = 1
    for i, tile in enumerate(raster_tiles):

        # Apply model to the tile
        imgsz_scale = W #544 # may want to change
        result = model(tile, conf=args.threshold, imgsz=imgsz_scale, device=device_name, verbose=False)

        # Save slice result
        if args.slices == "Ja" and len(result[0].boxes) != 0: # only if there is a detection
            result[0].save(filename=os.path.join(args.output,'output', 'slices', os.path.splitext(rastername)[0] + str(i+1) + '.png'), font_size=8, pil=True)

        # Get model's output
        if len(result[0].boxes) == 0:
            continue # no boxes were detected
        boxes = result[0].boxes.cpu().data.numpy() # [0] because only one image per iteration (can do batch)
        masks = masks = np.zeros((len(boxes), imgsz_scale, imgsz_scale), dtype=np.float32) # make empty masks object
        class_ids = result[0].boxes.cls.cpu().data.numpy()
        scores = result[0].boxes.conf.cpu().data.numpy()

        # Calculate the position of the tile in the original image
        x, y = tile_locations[i]

        # Update full mask if boxes were predicted
        if len(boxes) != 0:

            for instance in range(len(boxes)):
                # First convert box to segmentation mask
                x_min, y_min, x_max, y_max = boxes[instance][:4]
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
                masks[instance, y_min:y_max, x_min:x_max] = 1

                # Remove box if detected at the border
                # https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_clear_border.py
                # Use extra buffer (helps with preventing faulty 'close-to' border predictions of half birds) --> make dependent on the overlap
                if args.overlap <= 0.2:
                    buffer_s = 0
                else:
                    buffer_s = int(imgsz_scale/6)

                # But only do if the tile size is large enough (otherwise predictions always touch the border and will be removed)
                if args.grootte <= 2:
                    instance_mask = masks[instance]
                else:
                    instance_mask = clear_border(masks[instance], buffer_size=buffer_s)

                # Continue with next if mask was removed because of the border
                if np.all(instance_mask == 0) or np.all(instance_mask == False):
                    continue

                # Identify the pixels where the mask is true
                mask_true_indices = instance_mask == 1
                # mask_true_indices = masks[instance] == 1
                resized_mask = cv2.resize(mask_true_indices.astype(np.uint8), (W, W), interpolation=cv2.INTER_NEAREST)  # YOLOv8 makes mask of other size, so resize

                # Check for previous detection and confidence score of location first
                all_score_values = score_mask[x:x + W, y:y + W][resized_mask.astype(bool)]
                score_values = all_score_values[all_score_values != 0]

                # if more x% of the area already has a prediction (thus probably same object), check the score
                if (np.count_nonzero(all_score_values)/all_score_values.size) > 0.6:
                    unique_values, counts = np.unique(score_values, return_counts=True)
                    score_value = np.max(unique_values)
                    # score_value = unique_values[np.argmax(counts)]
                else:
                    score_value = 0

                # Update relevant portion of full_mask depending on confidence score
                if score_value < int(np.around(scores[instance]*100,0)):
                    # Make the prediction unique (prevent merging of two or more predictions)
                    if unique_n == 25:
                        unique_n = 1
                    unique_mask[x:x + W, y:y + W][resized_mask.astype(bool)] = float(str(int(class_ids[instance] + 1)) + '.' + str(unique_n))
                    unique_n = unique_n + 1

                    # Update the mask with shapes and their category
                    # full_mask[x:x + W, y:y + W][resized_mask.astype(bool)] = class_ids[instance] + 1  # +1 to leave 0 as background

                    # Update the score mask
                    score_mask[x:x + W, y:y + W][resized_mask.astype(bool)] = int(np.around(scores[instance]*100,0))

        # Save mask and progress
        if (i + 1) % 20 == 0:
            # Save mask as photo to see progress
            plt.imsave(os.path.join(args.output, 'output', os.path.splitext(rastername)[0]+"_YOLOv8_mask.png"), unique_mask)
            if args.backup == "Ja":
                np.save(os.path.join(args.output, 'output', os.path.splitext(rastername)[0]+"_YOLOv8_mask.npy"), unique_mask)
            # you can use the mask array to create shapefiles in case of crash/stop (continue at next step)

            # Calculate time
            end = time.time()
            time_consumed = end - start
            progress = len(raster_tiles) / (i + 1)
            total_time = time_consumed * progress # approximated

            # Print progress
            print("\r iteratie: {}/{}, tijd: {}/{}".format(i + 1, len(raster_tiles),
                                          time.strftime('%H:%M:%S', time.gmtime(int(time_consumed))),
                                          time.strftime('%H:%M:%S', time.gmtime(int(total_time))))) # , end='\r')


    print("Totale tijd:", time.strftime('%H:%M:%S', time.gmtime(int(time.time()-start)))) # Final time
    plt.imsave(os.path.join(args.output, 'output', os.path.splitext(rastername)[0]+"_YOLOv8_mask.png"), unique_mask) # Final mask save

# ==============================================================

# Convert mask to shapefile with classes

# ==============================================================

    print("-----------------------------------")
    print("Klaar, " + os.path.splitext(rastername)[0] + "_YOLOv8_mask.png" + " wordt geconverteerd naar ESRI shapefile...")
    print("-----------------------------------")

    # Set categories
    if os.path.isfile(os.path.join(str(args.model), 'model_categories.json')):
        with open(os.path.join(str(args.model), 'model_categories.json'), 'r') as json_file:
            categories = json.load(json_file) # make sure the categories are from the training and follow the indexing
    else:
        # This might be incorrect incase different/new species are added in training
        # categories = {'0': 'anders', '1': 'dwergmeeuw', '2': 'grote stern', '3': 'kluut', '4': 'kokmeeuw', '5': 'visdief', '6': 'zwartkopmeeuw'}
        print("Kon geen 'model_categories.json' bestand vinden in model folder...", "\n",
                 "Het model weet wel de soort index, maar niet bij welke index welke soort hoort... (wordt nu 'Unknown')")
        categories = {}

    # Apply function to create shapes (see helpers/mask2shape.py)
    mask_to_shape(args, unique_mask, categories, gt, "YOLOv8")

# Run
if __name__ == "__main__":
    main()
