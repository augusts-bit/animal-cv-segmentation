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

# YOLO and torch (check for correct versions)
import torch
from ultralytics import YOLO

# Parse arguments given to job
# import argparse

# import some common libraries
import numpy as np
import os, json, random
import re
import torch
import gc
import shutil
import time

# Other
import sys, os, distutils.core
from torchvision.io import read_image
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
import pandas as pd

# Install GDAL like this?
import subprocess
import sys

# Import helpers
# from helpers.def_custom_sliced_predict import *

# SAHI
from sahi.utils.yolov8 import (
    download_yolov8s_model,
)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
# from IPython.display import Image

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

def calculate_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_min1, y_min1, w1, h1 = bbox1
    x_min2, y_min2, w2, h2 = bbox2
    
    # Get the coordinates of the intersection rectangle
    x_left = max(x_min1, x_min2)
    y_top = max(y_min1, y_min2)
    x_right = min(x_min1 + w1, x_min2 + w2)
    y_bottom = min(y_min1 + h1, y_min2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate both bounding boxes areas
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute IoU
    return intersection_area / union_area

def apply_nms(coco_annotations, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to eliminate overlapping bounding boxes."""
    # Sort annotations by confidence score in descending order
    coco_annotations = sorted(coco_annotations, key=lambda x: x.get('score', 0), reverse=True)

    final_annotations = []

    while coco_annotations:
        # Take the annotation with the highest confidence
        current_annotation = coco_annotations.pop(0)
        current_bbox = current_annotation['bbox']
        final_annotations.append(current_annotation)

        # Compare this bbox with the rest and remove overlapping ones
        coco_annotations = [
            annotation for annotation in coco_annotations
            if calculate_iou(current_bbox, annotation['bbox']) < iou_threshold
        ]

    return final_annotations

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

# ==============================================================

# Inference/predict

# ==============================================================

    # Slicing Aided Hyper Inference (SAHI)
    # see https://github.com/obss/sahi/

    # Load detection model instance (from previously made weights and cfg) and prediction rasters
    # Check if a GPU is available
    print("-----------------------------------")
    print("cuda is available:", torch.cuda.is_available())
    print("-----------------------------------")
    if torch.cuda.is_available():
        # device_name = 'cuda:0'
        # GPU_MEM_LIMIT = (1024**3)*100 # 1024**3 is 1 GB memory limit
        # See C:\ProgramData\Anaconda3\envs\detectron2_env\lib\site-packages\detectron2\layers\mask_ops.py
        device_name = 'cpu' # cpu has more memory and is not much slower (?)
    else:
        device_name = 'cpu'

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
    
    # Tile pixel size (depending on cell size)
    rastername = os.path.basename(args.input)

    # Load detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=os.path.join(str(args.model), 'best.pt'), # or last.pt
        config_path=os.path.join(str(args.model), 'args.yaml'),
        confidence_threshold=args.threshold,
        category_mapping=categories,
        image_size=W, # YOLOv8 cannot accept 0 (to not resize like in Detectron2)
        device=device_name 
    )

    print("-----------------------------------")
    print("Loaded model, will start predicting")
    print("-----------------------------------")
    
    # May take long if large image
    result = get_sliced_prediction(
        raster_img, # image
        detection_model,
        # output_file_name = raster_name, 
        postprocess_type = "NMS",
        slice_height = W,
        slice_width = W,
        overlap_height_ratio = args.overlap,
        overlap_width_ratio = args.overlap,
        verbose = 2,
        perform_standard_pred = False
    )

# ==============================================================

# Convert to geodataframe/shapefile

# ==============================================================

    # To coco annotations
    coco_annotations = result.to_coco_annotations()
    coco_annotations = apply_nms(coco_annotations) # apply final nms

    # Save slice result
    if args.slices == "Ja":
        result.export_visuals(export_dir=os.path.join(args.output, 'output', 'slices'), hide_labels=True)

    # Geo-transform
    ds = gdal.Open(os.path.join(args.input))
    gt = ds.GetGeoTransform()  # Get geotransform information

    # Initialize a list to store the converted polygons
    geographic_polygons = []
    soort_values = []
    confidences = []

    for annotation in coco_annotations:
        bbox = annotation['bbox']  # Get bounding box [x_min, y_min, width, height]
        category_name = annotation.get('category_name', "Unclear")
        confidence = annotation.get('score', None)
        
        # Extract bounding box coordinates
        x_min, y_min, width, height = bbox

        # Calculate the four corners of the bounding box
        top_left = [x_min, y_min]
        top_right = [x_min + width, y_min]
        bottom_right = [x_min + width, y_min + height]
        bottom_left = [x_min, y_min + height]

        # Create a polygon from these points (in pixel coordinates)
        polygon = [top_left, top_right, bottom_right, bottom_left, top_left]

        # Initialize a list to store the converted points (geographic coordinates)
        geographic_points = []

        # Loop through the polygon points to convert to geographic coordinates
        for x_pixel, y_pixel in polygon:
            x_geo = gt[0] + (x_pixel * gt[1]) + (y_pixel * gt[2])
            y_geo = gt[3] + (x_pixel * gt[4]) + (y_pixel * gt[5])
            geographic_points.append([x_geo, y_geo])

        # Add the converted points to the list of polygons
        geographic_polygons.append(geographic_points)
        soort_values.append(category_name)
        confidences.append(confidence)

    # Delete gt, ds
    del result, gt, ds
    torch.cuda.empty_cache()

    # Create geodataframe from polygons
    gdf = gpd.GeoDataFrame(geometry=[Polygon(polygon) for polygon in geographic_polygons])
    gdf['pred_id'] = gdf.index + 1
    gdf['soort'] = soort_values
    gdf['confidence'] = confidences

    # Filter out polygons with invalid geometries
    gdf_pp = gdf[gdf.geometry.is_valid]

    # Filter polygons smaller than a certain area
    min_area_threshold = 1e-2  # Adjust this value based on your specific needs

    # Filter out polygons with area below the threshold
    gdf_pp = gdf_pp[gdf_pp.area >= min_area_threshold]

# ==============================================================

# Export

# ==============================================================

    print("-----------------------------------")
    print("Exporting...")
    print("-----------------------------------")

    # Add reference
    with rasterio.open(args.input) as src:
        crs = src.crs
    gdf_pp = gdf_pp.set_crs(src.crs)

    # Add column with nearest distance
    gdf_pp_joined = gpd.sjoin_nearest(gdf_pp, gdf_pp, distance_col="k_afstand", exclusive=True) #.reset_index(drop=True)
    gdf_pp_joined = gdf_pp_joined.rename(columns={'pred_id_left': 'pred_id'})
    gdf_pp = gdf_pp.merge(gdf_pp_joined[["pred_id", "k_afstand"]], on='pred_id')

    # And area
    gdf_pp['grootte'] = gdf_pp.area

    # Save
    gdf_pp.to_file(os.path.join(args.output, "output", os.path.splitext(rastername)[0] + "_polygons.shp"))

    # Delete objects
    del geographic_polygons, gdf, gdf_pp
    torch.cuda.empty_cache()
    gc.collect()

    print("-----------------------------------")
    print("Finished!")
    print("-----------------------------------")

# Run
if __name__ == "__main__":
    main()