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

# Parse arguments given to job
# import argparse

# import some common libraries
import numpy as np
import os, json, random
# import cv2
# from google.colab.patches import cv2_imshow
import re
import torch
# import tensorboard
# import tabulate

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

# ==============================================================

# Convert rasters to images, use subsets only if needed

# ==============================================================

# Convert dictionary to list
# rnames = list(args.rasternames.values())

# List tif files and their names
rnames = [os.path.splitext(f)[0] for f in os.listdir('input') if f.lower().endswith(('.tif', '.tiff'))]
print("-----------------------------------")
print("Rasters", rnames)
print("-----------------------------------")

# Rasters
test_folder = 'input' # Folder containing rasters for prediction
rasters = rnames # E.g., ['steenplaat_vlucht1', 'waterdunen_vlucht2']
create_subset = "no" # (SAHI cannot handle input that is too large)

print("-----------------------------------")
print("Will create subsets:", create_subset)
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

# Rasters may have different amount of subsets
subset_counts = []

# Create subsets (SAHI cannot handle input that is too large)
if create_subset == "yes":

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

            # how divide width and height
            if src.width < 3000:
                subset_n = 1
            else:
                subset_n = 10  # may or may not be enough (too large subsets may cause OOM)
                subset_n = round(src.width / 2000)  # 10 Waterdunen subsets (of width ~3000). Was more or less good.

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

            # Append subset count
            subset_counts.append(subset_count)

        # Delete for memory
        del src, dst
        torch.cuda.empty_cache()

        # Stop looping if only one item
        if len(rasters) == 1:
            break

    print("-----------------------------------")
    print("Created subsets")
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
    torch.cuda.empty_cache()
    return(raster_img)

# store (sub)raster names
raster_names = []

if create_subset == "no":
    subset_counts = [0]

for raster, subset_count in zip(rasters, subset_counts):

    # Check if there is only one item (input)
    if len(rasters) == 1:
        raster = rasters[0]

    if create_subset == "yes":

        for n in range(1, subset_count):

            if os.path.isfile("output/subsets/" + raster + "_subset" + "_" + str(n) + ".tif"):
                    # Open saved raster
                    raster_data_set = gdal.Open("output/subsets/" + raster + "_subset" + "_" + str(n) + ".tif")
            else:
                    continue

            # Save time and memory by removing bad rasters
            stacked_tif = readtif(raster_data_set)

            # Check if a certain percentage is black or white pixels and if so filter out
            # if np.all(stacked_tif != 0) or np.sum(stacked_tif) >= 0.9*255:
            if ((np.sum(stacked_tif == 0) / stacked_tif.size) * 100) < 5: # or ((np.sum(stacked_tif >= 255) / stacked_tif.size) * 100) < 50:

                out_img = "output/rasterimages/" + raster + "_subset" + "_" + str(n) + ".png"
                out_paths.append(out_img)
                raster_names.append(raster + "_subset" + "_" + str(n))

                # Save image to file
                plt.imsave(out_img, stacked_tif.astype(np.uint8), cmap='gray', format='png')  # Convert to uint8

            # Delete for memory
            del stacked_tif
            torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()

    # Stop looping if only one item
    if len(rasters) == 1:
        break

print("-----------------------------------")
print("Converted rasters to images (see output/rasterimages)")
print("-----------------------------------")
