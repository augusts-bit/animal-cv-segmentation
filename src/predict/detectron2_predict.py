# ==============================================================

# Setup and import

# ==============================================================

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Detectron and torch (check for correct versions)
import torch, detectron2

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
import time

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

# SAHI
# from sahi.utils.detectron2 import export_cfg_as_yaml
# from sahi.utils.detectron2 import Detectron2TestConstants
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction, predict, get_prediction
# from sahi.utils.file import download_from_url
# from sahi.utils.cv import read_image

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

# Start

# ==============================================================

def main(args):
    
    # First create output directory
    if os.path.isdir(os.path.join(args.output, 'output')) == False:
        os.makedirs(os.path.join(args.output, 'output'))

    print("-----------------------------------")
    print("Alle output bevindt zicht in "+str(os.path.join(args.output,'output')))
    print("-----------------------------------")

# ==============================================================

# Load image / raster and tile

# ==============================================================

    print("-----------------------------------")
    print(args.input, "wordt geknipt... (zie slices)")
    print("-----------------------------------")

    # Save folder
    if os.path.isdir(os.path.join(args.output, 'output',
                                  'slices')):  # remove old slices folder so that it always contains slices of current run (will be made during prediction)
        shutil.rmtree(os.path.join(args.output, 'output', 'slices'))

    os.makedirs(os.path.join(args.output, 'output', 'slices'))

    # Load raster, convert to image and tile
    rastername = os.path.basename(args.input)
    raster_data_set = gdal.Open(args.input)

    gt = raster_data_set.GetGeoTransform()
    pixelSizeX = gt[1]  # cell size

    # Convert to array image
    raster_band_1 = raster_data_set.GetRasterBand(1)  # red channel
    raster_band_2 = raster_data_set.GetRasterBand(2)  # green channel
    raster_band_3 = raster_data_set.GetRasterBand(3)  # blue channel
    raster_b1 = raster_band_1.ReadAsArray()
    raster_b2 = raster_band_2.ReadAsArray()
    raster_b3 = raster_band_3.ReadAsArray()

    # Stack to image
    raster_img = np.dstack((raster_b1, raster_b2, raster_b3))

    # Tile pixel size (depending on cell size)
    W = int(args.grootte / pixelSizeX)
    # target_cs = 0.011272670412636411 # cell size of Texel PHZ
    # W = round(445 * (target_cs/pixelSizeX)) # ~ 5.0 m x 5.0 m # was 350

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
        raster_tile_name = os.path.join(args.output, 'output', 'slices', os.path.splitext(rastername)[0] + str(n) + ".png")
        raster_tile = cv2.convertScaleAbs(raster_tile)  # convert to uint8
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
    cfg.merge_from_file(os.path.join(str(args.model),'model_cfg.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(str(args.model),'model_weights.pt')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold   # set a custom testing threshold
    cfg.INPUT.MIN_SIZE_TEST = 538 # 0 is no rescale
    cfg.INPUT.MIN_SIZE_TRAIN = (538,) # trained on 538
    cfg.MODEL.DEVICE = device_name
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000 # Detect more instances? (https://github.com/facebookresearch/detectron2/issues/1481)
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
    predictor = DefaultPredictor(cfg)

    print("-----------------------------------")
    print("Model is geladen, begint met segmentatie en classificatie...", "\n"
          "Zie " + str(os.path.splitext(rastername)[0]) + "_MaskRCNN_mask.png voor progressie", "\n")

    if args.backup == "Ja":
        print("Gebruik " + str(os.path.splitext(rastername)[0]) + "_MaskRCNN_mask.npy voor back-up (met predict_from_mask.py)")
    print("-----------------------------------")

    # Create masks to keep track of predictions for the entire original image
    score_mask = np.zeros_like(raster_img[:, :, 0], dtype=np.uint8)
    unique_mask = np.zeros_like(raster_img[:, :, 0],
                                dtype=np.float32)  # float, store category as the number with a unique decimal

    # unique mask replaces full_mask
    # full_mask = np.zeros_like(raster_img[:,:,0], dtype=np.uint8)

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
    start = time.time()  # Check how long it takes
    unique_n = 1
    for i, tile in enumerate(raster_tiles):
    # for i, tile in tqdm(enumerate(raster_tiles), total=len(raster_tiles)):

        # Load tile image
        tile_path = os.path.join(args.output, 'output', 'slices', os.path.splitext(rastername)[0] + str(i+1) + ".png")
        im = cv2.imread(tile_path)

        # Skip if white image
        if all_white_pixels(im):
            continue

        # Apply model to the tile
        outputs = predictor(im)

        # Get model's output
        masks = outputs["instances"].get('pred_masks').cpu().numpy()
        class_ids = outputs["instances"].get('pred_classes').cpu().numpy()
        scores = outputs["instances"].get('scores').cpu().numpy()

        # Calculate the position of the tile in the original image
        x, y = tile_locations[i]

        # Update full mask if masks were predicted
        if len(masks) != 0:
            # Sum the masks along the first axis before combining with full_mask
            combined_mask = np.sum(masks, axis=0)

            for instance in range(len(masks)):
                # Remove mask if detected at the border
                # https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_clear_border.py
                # Use extra buffer (helps with preventing faulty 'close-to' border predictions of half birds) --> make dependent on the overlap
                if args.overlap <= 0.2:
                    buffer_s = 0
                else:
                    buffer_s = int(cfg.INPUT.MIN_SIZE_TEST / 6)

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

                # Check for previous detection and confidence score of location first
                all_score_values = score_mask[x:x + W, y:y + W][mask_true_indices]
                score_values = all_score_values[all_score_values != 0]

                # if more x% of the area already has a prediction (thus probably same object), check the score
                if (np.count_nonzero(all_score_values) / all_score_values.size) > 0.6:
                    unique_values, counts = np.unique(score_values, return_counts=True)
                    score_value = np.max(unique_values)
                    # score_value = unique_values[np.argmax(counts)]
                else:
                    score_value = 0

                # Update relevant portion of full_mask depending on confidence score
                if score_value < int(np.around(scores[instance] * 100, 0)):
                    # Make the prediction unique (prevent merging of two or more predictions)
                    if unique_n == 25:
                        unique_n = 1
                    unique_mask[x:x + W, y:y + W][mask_true_indices] = float(
                        str(int(class_ids[instance] + 1)) + '.' + str(unique_n))
                    unique_n = unique_n + 1

                    # Update the mask with shapes and their category
                    # full_mask[x:x + W, y:y + W][mask_true_indices] = class_ids[instance] + 1  # +1 to leave 0 as background

                    # Update the score mask
                    score_mask[x:x + W, y:y + W][mask_true_indices] = int(np.around(scores[instance] * 100, 0))

        # Save mask and progress
        if (i + 1) % 20 == 0:
            # Save mask as photo to see progress
            plt.imsave(os.path.join(args.output, 'output', os.path.splitext(rastername)[0] + "_MaskRCNN_mask.png"),
                       unique_mask)
            if args.backup == "Ja":
                np.save(os.path.join(args.output, 'output', os.path.splitext(rastername)[0] + "_MaskRCNN_mask.npy"),
                    unique_mask)
            # you can use the mask array to create shapefiles in case of crash/stop (continue at next step)

            # Calculate time
            end = time.time()
            time_consumed = end - start
            progress = len(raster_tiles) / (i + 1)
            total_time = time_consumed * progress  # approximated

            # Print progress
            print("\r iteratie: {}/{}, tijd: {}/{}".format(i + 1, len(raster_tiles),
                                                           time.strftime('%H:%M:%S', time.gmtime(int(time_consumed))),
                                                           time.strftime('%H:%M:%S',
                                                                         time.gmtime(int(total_time)))))  # , end='\r')

    print("Totale tijd:", time.strftime('%H:%M:%S', time.gmtime(int(time.time() - start))))  # Final time
    plt.imsave(os.path.join(args.output, 'output', os.path.splitext(rastername)[0] + "_MaskRCNN_mask.png"),
               unique_mask)  # Final mask save

    # ==============================================================

    # Convert mask to shapefile with classes

    # ==============================================================

    print("-----------------------------------")
    print("Klaar, " + os.path.splitext(rastername)[
        0] + "_YOLOv8_mask.png" + " wordt geconverteerd naar ESRI shapefile...")
    print("-----------------------------------")

    # Set categories
    if os.path.isfile(os.path.join(str(args.model), 'model_categories.json')):
        with open(os.path.join(str(args.model), 'model_categories.json'), 'r') as json_file:
            categories = json.load(json_file)  # make sure the categories are from the training and follow the indexing
    else:
        # This might be incorrect incase different/new species are added in training
        # categories = {'0': 'anders', '1': 'dwergmeeuw', '2': 'grote stern', '3': 'kluut', '4': 'kokmeeuw', '5': 'visdief', '6': 'zwartkopmeeuw'}
        print("Kon geen 'model_categories.json' bestand vinden in model folder...", "\n",
              "Het model weet wel de soort index, maar niet bij welke index welke soort hoort... (wordt nu 'Unknown')")
        categories = {}

    # Apply function to create shapes (see helpers/mask2shape.py)
    mask_to_shape(args, unique_mask, categories, gt, "MaskRCNN")

# Run
if __name__ == "__main__":
    main()