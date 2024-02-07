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
print("imported torch and YOLOv8")

# Parse arguments given to job
# import argparse

# import some common libraries
import numpy as np
import os, json, random
import re
import torch
import gc
import shutil

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

# Install GDAL like this?
import subprocess
import sys

# Import helpers
from helpers.def_custom_sliced_predict import *

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

# Inference/predict

# ==============================================================

# Slicing Aided Hyper Inference (SAHI)
# see https://github.com/obss/sahi/

# Define what images we want to predict
create_subset = "no" # were the input rasters subsetted?
# (rasters that were convered into) images for prediction (only those in the main folder)
raster_names = [os.path.splitext(file)[0] for file in os.listdir("output/rasterimages") if os.path.isfile(os.path.join("output/rasterimages", file))]
out_paths = [os.path.join('output', 'rasterimages', f'{item}.png') for item in raster_names]

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
if os.path.isfile('model/yolov8/model_categories.json'):
    with open('model/yolov8/model_categories.json', 'r') as json_file:
        categories = json.load(json_file) # make sure the categories are from the training and follow the indexing
else:
    # This might be incorrect incase different/new species are added in training
    categories = {'0': 'anders', '1': 'dwergmeeuw', '2': 'grote stern', '3': 'kluut', '4': 'kokmeeuw', '5': 'visdief', '6': 'zwartkopmeeuw'}

# Load detection model from trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="model/yolov8/best.pt", # or last.pt
    config_path="model/yolov8/args.yaml",
    confidence_threshold=0.5,
    category_mapping=categories,
    image_size=544, # YOLOv8 cannot accept 0 (to not resize like in Detectron2), must be multiple of max stride 32 (trained on 538, so updating to 544)
    device=device_name #'cuda:0' # "cpu"
)

print("-----------------------------------")
print("Loaded model, will start predicting")
print("-----------------------------------")

# Make prediction
# Loop through specified rasters (now images) to make predictions
for out_img, raster_name in zip(out_paths, raster_names):

    # Check if there is only one item (input)
    if len(out_paths) == 1:
        out_img = out_paths[0]
        raster_name = raster_names[0]

    print("-----------------------------------")
    print("Starting with " + raster_name)
    print("-----------------------------------")

    # Base slices of tile size during training
    # target_cs = 0.011272670412636411  # cell size of Texel PHZ
    target_cs = 0.005600650546195552 # cell size of Wagejot 2022

    if create_subset == "yes":
        raster_data_set = gdal.Open("output/subsets/" + raster_name + ".tif")
    else:
        raster_data_set = gdal.Open("input/" + raster_name + ".tif")

    # 10m ~ round(890 * (target_cs/pixel_size)), 5m ~ slice_size = round(445 * (target_cs / pixel_size)) --> training (Texel PHZ)
    pixel_size = raster_data_set.GetGeoTransform()[1]
    slice_size = round(500 * (target_cs / pixel_size))  # make smaller than what it's trained on (?)

    # remove old slices folder so that it always contains slices of current run (will be made during prediction)
    if os.path.isdir("slices"):
        shutil.rmtree("slices")

    # May take long if large image
    result = custom_get_sliced_prediction(
        out_img, # path to image
        detection_model,
        output_file_name = raster_name, # save slices
        slice_height = slice_size,
        slice_width = slice_size,
        overlap_height_ratio = 0.5,
        overlap_width_ratio = 0.5,
        verbose = 2,
    )

    # os.remove(os.path.join("output/visual", "prediction_visual.png"))
    result.export_visuals(export_dir="output/visual", hide_labels=True)

    if create_subset == "yes":
        # Rename exported result (standard name = "prediction_visual.png")
        old_file = os.path.join("output/visual", "prediction_visual.png")
        new_file = os.path.join("output/visual", raster_name + "_visual.png")

        # Remove file first if name exists (or error)
        if os.path.exists(new_file):
            os.remove(new_file)
        os.rename(old_file, new_file)
    else:
        old_file = os.path.join("output/visual", "prediction_visual.png")
        new_file = os.path.join("output/visual", raster_name + "_visual.png")

        if os.path.exists(new_file):
            os.remove(new_file)
        os.rename(old_file, new_file)

    print("-----------------------------------")
    print("Made prediction on " + raster_name + " (see output/visual)")
    print("-----------------------------------")

    # To coco annotations
    coco_annotations = result.to_coco_annotations()

    # Geo-transform
    if create_subset == "yes":

        ds = gdal.Open(os.path.join("output/subsets/", raster_name + ".tif"))
        gt = ds.GetGeoTransform()  # Get geotransform information

        # gt will contain information like pixel size, rotation, and top-left corner coordinates
        # print(f"GeoTransform: {gt}")

    else:

        ds = gdal.Open(os.path.join('input', raster_name + ".tif"))
        gt = ds.GetGeoTransform()  # Get geotransform information

    # Initialize a list to store the converted polygons
    geographic_polygons = []
    soort_values = []

    # Loop through the COCO annotations
    for annotation in coco_annotations:
        segmentation = annotation['segmentation'][0]  # Assuming only one segmentation per detected object
        category_name = annotation.get('category_name', "Unclear")

        # Initialize a list to store the converted points
        geographic_points = []

        # Loop through the segmentation points
        for x_pixel, y_pixel in zip(segmentation[::2], segmentation[1::2]):
            x_geo = gt[0] + (x_pixel * gt[1]) + (y_pixel * gt[2])
            y_geo = gt[3] + (x_pixel * gt[4]) + (y_pixel * gt[5])
            geographic_points.append([x_geo, y_geo])

        # Add the converted points to the list of polygons
        geographic_polygons.append(geographic_points)
        soort_values.append(category_name)

    # Delete gt, ds
    del result, gt, ds
    torch.cuda.empty_cache()

    # 'geographic_polygons' now contains a list of polygons in geographic coordinates
    # Create geodataframe from polygons
    gdf = gpd.GeoDataFrame(geometry=[Polygon(polygon) for polygon in geographic_polygons])
    gdf['soort'] = soort_values

    # Post-process
    print("-----------------------------------")
    print("Prediction:", raster_name)

    # Create a copy of the GeoDataFrame for safe processing
    gdf_pp = gdf.copy()
    print("Number of shapes originally:", len(gdf_pp))

    # Filter out polygons with invalid geometries
    gdf_pp = gdf_pp[gdf_pp.geometry.is_valid]
    print("After removing invalid geometries:", len(gdf_pp))

    # Remove polygons that are within others
    gdf_pp['within'] = False
    buffer = 1e-1  # Buffer (otherwise some/most still stay)

    # Create bounding boxes for each geometry (technically not necessary as we already had bboxes)
    gdf_pp['bbox'] = gdf_pp.geometry.apply(lambda geom: geom.bounds)

    # Iterate over bounding boxes
    for index_i, row_i in gdf_pp.iterrows():
        minx_i, miny_i, maxx_i, maxy_i = row_i['bbox']

        for index_j, row_j in gdf_pp.iterrows():

            minx_j, miny_j, maxx_j, maxy_j = row_j['bbox']

            if index_i != index_j:
                if minx_i >= minx_j - buffer and miny_i >= miny_j - buffer and maxx_i <= maxx_j + buffer and maxy_i <= maxy_j + buffer:
                    gdf_pp.at[index_i, 'within'] = True

    # Filter rows
    gdf_pp = gdf_pp.loc[gdf_pp['within'] == False]

    # Drop columns so only segmentation polygons are left
    gdf_pp = gdf_pp.drop(columns=['bbox', 'within'])

    print("After dropping smaller shapes that overlap:", len(gdf_pp))

    # Filter polygons smaller than a certain area
    min_area_threshold = 1e-2  # Adjust this value based on your specific needs

    # Filter out polygons with area below the threshold
    gdf_pp = gdf_pp[gdf_pp.area >= min_area_threshold]

    print("After dropping shapes that are too small:", len(gdf_pp))
    print("-----------------------------------")

    # Save
    if create_subset == "yes":
        with rasterio.open("output/subsets/" + raster_name + ".tif") as src:
            gdf_pp = gdf_pp.set_crs(src.crs)
            gdf_pp.to_file(os.path.join("output/prediction/", raster_name + "_polygons.shp"))
    else:
        with rasterio.open(os.path.join('input', raster_name + ".tif")) as src:
            gdf_pp = gdf_pp.set_crs(src.crs)
            gdf_pp.to_file(os.path.join("output/prediction/", raster_name + "_polygons.shp"))

    print("-----------------------------------")
    print("Done for " + raster_name + "! (see output/prediction)")
    print("-----------------------------------")

    # Delete objects
    del geographic_polygons, gdf, gdf_pp
    torch.cuda.empty_cache()
    gc.collect()

    # Stop looping if only one item
    if len(out_paths) == 1:
        break

# Remove subset rasters (take much space and are not necessary anymore)
if os.path.isdir("output/subsets"):
    shutil.rmtree("output/subsets")

print("-----------------------------------")
print("Finished!")
print("-----------------------------------")

