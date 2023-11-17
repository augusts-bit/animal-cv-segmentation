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
import os, json, random
# import cv2
# from google.colab.patches import cv2_imshow
import re
import torch
# import tensorboard
# import tabulate

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
# from IPython.display import Image

# Other
import sys, os, distutils.core
# from PIL import Image
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2
from osgeo import gdal
import geopandas as gpd
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

# Install GDAL like this?
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdal"])

from osgeo import gdal
# import gdal

# Log metrics
# import mlflow
# import mlflow.pytorch

# ==============================================================

# Load inputs given to job

# ==============================================================

# This is how you load input from job
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-train-model?view=azureml-api-2 

# input and output arguments
# parser = argparse.ArgumentParser()

# Rasters
# parser.add_argument("--rasterdata")
# parser.add_argument("--rasternames")

# Create subset?
# parser.add_argument("--create_subset")
# args = parser.parse_args()

# ==============================================================

# Convert rasters to images, use subsets only if needed

# ==============================================================

# Convert dictionary to list
# rnames = list(args.rasternames.values())

# List tif files and their names
rnames = [os.path.splitext(f)[0] for f in os.listdir('input') if f.lower().endswith(('.tif', '.tiff'))]
print("-----------------------------------")
print("Prediction on", rnames)
print("-----------------------------------")

# Rasters
test_folder = 'input' # Folder containing rasters for prediction
rasters = rnames # E.g., ['steenplaat_vlucht1', 'waterdunen_vlucht2']
create_subset = "yes" # (SAHI cannot handle input that is too large)

print("-----------------------------------")
print("Will do predictions on subsets:", create_subset)
print("-----------------------------------")

# Output folders
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

create_directory_if_not_exists("output")
create_directory_if_not_exists("output/subsets")
create_directory_if_not_exists("output/rasterimages")
create_directory_if_not_exists("output/postprocessing")
create_directory_if_not_exists("output/visual")
create_directory_if_not_exists("output/prediction")

print("-----------------------------------")
print("Created output folders")
print("-----------------------------------")

# Create subsets (SAHI cannot handle input that is too large)
if create_subset == "yes":

    # how divide width and height
    subset_n = 4  # may or may not be enough

    for raster in rasters:

        # Check if there is only one item (input) --> prevent looping over characters
        if len(rasters) == 1:
            raster = rasters[0]

        # Read path
        raster_data_path = os.path.join(test_folder, raster + ".tif")
        out_path = "output/subsets/" + raster + "_subset"

        # Open the original raster file
        with rasterio.open(raster_data_path) as src:
            # Get the dimensions of the original raster
            width = src.width
            height = src.height

            # Make subset raster (loop over width and height separately)
            subset_count = 1
            for n in range(subset_n):

                for m in range(subset_n):

                    # Calculate the bounding box coordinates (check what gives nice subset)
                    xmin = (width / subset_n) * n
                    ymin = (height / subset_n) * m
                    xmax = (width / subset_n) * (n+1)
                    ymax = (height / subset_n) * (m+1)
                    # xmin = width // 10
                    # ymin = height // 2
                    # xmax = width // 3
                    # ymax = height // 1.5

                    # print(xmin, ymin, xmax, ymax)

                    # Define a window using these coordinates
                    window = Window(xmin, ymin, xmax - xmin, ymax - ymin)

                    # Read the data within the window
                    quarter_data = src.read(window=window)

                    # Update the transform to reflect the new window
                    new_transform = src.window_transform(window)

                    out_subset_path = out_path + "_" + str(subset_count) + ".tif"
                    subset_count = subset_count+1

                    # Create a new raster file with the subset data
                    with rasterio.open(out_subset_path, 'w', driver='GTiff',
                            width=xmax - xmin, height=ymax - ymin, count=src.count,
                            dtype=src.dtypes[0], crs=src.crs, transform=new_transform) as dst:
                                dst.write(quarter_data)

        # Stop looping if only one item
        if len(rasters) == 1:
            break

    print("-----------------------------------")
    print("Created subsets (see output/subsets)")
    print("-----------------------------------")

if create_subset != "yes" and create_subset != "no":
    sys.exit("Please specify if you want to create subsets ('yes' or 'no').")

else:
    pass

# Convert tif to img
out_paths = []

def readtif(tif):
    # Read
    raster_band_1 = tif.GetRasterBand(1) # red channel
    raster_band_2 = tif.GetRasterBand(2) # green channel
    raster_band_3 = tif.GetRasterBand(3) # blue channel

    # Convert to array image
    raster_b1 = raster_band_1.ReadAsArray()
    raster_b2 = raster_band_2.ReadAsArray()
    raster_b3 = raster_band_3.ReadAsArray()

    # Stack to image
    raster_img = np.dstack((raster_b1, raster_b2, raster_b3))
    del raster_band_1, raster_band_2, raster_band_3, raster_b1, raster_b2, raster_b3
    return(raster_img)

# store (sub)raster names
raster_names = []

for raster in rasters:

    # Check if there is only one item (input)
    if len(rasters) == 1:
        raster = rasters[0]

    if create_subset == "yes":

        for n in range(1, subset_count):

            # Open saved raster
            raster_data_set = gdal.Open("output/subsets/" + raster + "_subset" + "_" + str(n) + ".tif")

            # Save time and memory by removing bad rasters
            stacked_tif = readtif(raster_data_set)

            # Check if not majority black or white pixels and if so filter out
            # if np.all(stacked_tif != 0) or np.sum(stacked_tif) >= 0.9*255:
            if ((np.sum(stacked_tif == 0) / stacked_tif.size) * 100) < 50: # or ((np.sum(stacked_tif >= 255) / stacked_tif.size) * 100) < 50:

                out_img = "output/rasterimages/" + raster + "_subset" + "_" + str(n) + ".png"
                out_paths.append(out_img)
                raster_names.append(raster + "_subset" + "_" + str(n))

                # Save image to file
                plt.imsave(out_img, stacked_tif.astype(np.uint8), cmap='gray', format='png')  # Convert to uint8

            # Delete for memory
            del stacked_tif

    else:
        
        # raster_data_set = gdal.Open(os.path.join(test_folder, raster) + ".tif")

        # Check if .tif extension is lower or upper case
        tif_path = os.path.join(test_folder, raster + ".tif")
        if os.path.exists(tif_path):
            raster_data_set = gdal.Open(tif_path)
        else:
            # Check if .TIF file exists
            tif_path_upper = os.path.join(test_folder, raster + ".TIF")
            if os.path.exists(tif_path_upper):
                raster_data_set = gdal.Open(tif_path_upper)
            else:
                # Both .tif and .TIF files don't exist
                print(f"No file found for {raster}")

        out_img = "output/rasterimages/" + raster + ".png"
        out_paths.append(out_img)
        raster_names.append(raster)

        # Save to file
        stacked_tif = readtif(raster_data_set)
        plt.imsave(out_img, stacked_tif.astype(np.uint8), cmap='gray', format='png')  # Convert to uint8

        # Delete for memory
        del stacked_tif

    # Stop looping if only one item
    if len(rasters) == 1:
        break

print("-----------------------------------")
print("Converted rasters to images (see output/rasterimages)")
print("-----------------------------------")

# ==============================================================

# Inference/predict

# ==============================================================

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator("vali", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "vali")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`

# Slicing Aided Hyper Inference (SAHI)
# see https://github.com/obss/sahi/

# Load detection model instance (from previously made weights and cfg) and prediction rasters

# Check if a GPU is available
print("-----------------------------------")
print("cuda is available:", torch.cuda.is_available())
print("-----------------------------------")
if torch.cuda.is_available():
    device_name = 'cuda:0' 
else:
    device_name = 'cpu'

# GPU_MEM_LIMIT = (1024**3)*100 # 1024**3 is 1 GB memory limit
# See C:\ProgramData\Anaconda3\envs\detectron2_env\lib\site-packages\detectron2\layers\mask_ops.py

# Load detection model from trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path="model/model_weights.pt", 
    config_path="model/model_cfg.yaml", 
    confidence_threshold=0.3,
    image_size=640, # (does this matter?)
    device=device_name #'cuda:0' # "cpu"
)

print("-----------------------------------")
print("Loaded model, will start predicting")
print("-----------------------------------")

# Make prediction
import def_custom_sliced_predict # SAHI predict function but with print statement

# Loop through specified rasters (now images) to make predictions
for out_img, raster_name in zip(out_paths, raster_names):

    # Check if there is only one item (input)
    if len(out_paths) == 1:
        out_img = out_paths[0]
        raster_name = raster_names[0]

    print("-----------------------------------")
    print("Starting with " + raster_name)
    print("-----------------------------------")

    # Takes very long if large image
    result = def_custom_sliced_predict.custom_get_sliced_prediction(
        out_img, # path to image
        detection_model,
        slice_height = 800, # test what suffices (assumes photos are taken with same height -- make height dependent variable ?)
        slice_width = 800, # i.d.
        overlap_height_ratio = 0, # The higher the overlap, the less it misses (?)
        overlap_width_ratio = 0,
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

        ds = gdal.Open(os.path.join(test_folder, raster_name + ".tif"))
        gt = ds.GetGeoTransform()  # Get geotransform information

    # Initialize a list to store the converted polygons
    geographic_polygons = []

    # Loop through the COCO annotations
    for annotation in coco_annotations:
        segmentation = annotation['segmentation'][0]  # Assuming only one segmentation per detected object

        # Initialize a list to store the converted points
        geographic_points = []

        # Loop through the segmentation points
        for x_pixel, y_pixel in zip(segmentation[::2], segmentation[1::2]):
            x_geo = gt[0] + (x_pixel * gt[1]) + (y_pixel * gt[2])
            y_geo = gt[3] + (x_pixel * gt[4]) + (y_pixel * gt[5])
            geographic_points.append([x_geo, y_geo])

        # Add the converted points to the list of polygons
        geographic_polygons.append(geographic_points)

    # 'geographic_polygons' now contains a list of polygons in geographic coordinates
    # Create geodataframe from polygons
    gdf = gpd.GeoDataFrame(geometry=[Polygon(polygon) for polygon in geographic_polygons])

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
            raster_crs = src.crs
            gdf_pp.crs = raster_crs
            gdf_pp.to_file(os.path.join("output/prediction/", raster_name + "_polygons.shp"))
    else:
        with rasterio.open(os.path.join(test_folder, raster_name + ".tif")) as src:
            raster_crs = src.crs
            gdf_pp.crs = raster_crs
            gdf_pp.to_file(os.path.join("output/prediction/", raster_name + "_polygons.shp"))

    print("-----------------------------------")
    print("Done for " + raster_name + "! (see output/prediction)")
    print("-----------------------------------")


    # Stop looping if only one item
    if len(out_paths) == 1:
        break

print("-----------------------------------")
print("Finished!")
print("-----------------------------------")

