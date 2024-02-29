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

# Function to create a shapefile from a (float) mask
def mask_to_shape(args, unique_mask, categories, gt, model_name, frommask=False):

    # Get rastername
    if frommask == False: # rastername variable is stored in 'args.input'
        rasterloc = args.input
        rastername = os.path.basename(args.input)
    else:
        rasterloc = args.raster # rastername variable is stored in 'args.input'
        rastername = os.path.basename(args.raster)

    # Identify contours for each bird and what species they are
    contours = []
    majority_values = []

    for class_value in np.unique(unique_mask): # np.unique(full_mask)
        if class_value < 1:  # Skip background
            continue

        # If mask is of uint8 (no float)
        # mask_class = (full_mask == class_value).astype(np.uint8)
        # contours_class, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_class = np.where(unique_mask == class_value, 1, 0).astype(np.uint8)
        contours_class, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_class:
            # Find the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region of interest (ROI)
            roi = unique_mask[y:y + h, x:x + w]

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
            majority_values.append(math.floor(class_value)) # class_value should always be majority value
            # majority_values.append(majority_value) # first did this

    # Create GeoDataFrame to store polygons
    polygons = gpd.GeoDataFrame(columns=['pred_id', 'soort_id', 'soort', 'grootte', 'geometry'])
    id = 1

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
        instance_row = pd.DataFrame({'pred_id': id, 'soort_id': majority_value, 'soort': species_name, 'grootte': polygon.area, 'geometry': polygon}, index=[0])
        polygons = pd.concat([polygons, instance_row], ignore_index=True)
        id = id + 1

    # Remove geometries smaller than 0.005 (small leftover pixel masks due to merging or borders)
    polygons = polygons[polygons['grootte'] >= 0.005] # may need to determine threshold with ArcGIS Pro/QGIS

    # Transform geometry
    with rasterio.open(rasterloc) as src:
        transform = src.transform
        crs = src.crs

    # Transform the geometry of the polygons to match that of the original raster (may still need to do 'Define Projection' in ArcGIS Pro)
    polygons = polygons.set_crs(crs)

    # Join nearest to spot bad predictions (e.g., double predictions for one bird are often really close to each other)
    if sys.version_info >= (3, 9): # can only with Python >=3.9 for right GeoPandas version
        polygons_joined = gpd.sjoin_nearest(polygons, polygons, distance_col="k_afstand", exclusive=True) #.reset_index(drop=True)
        polygons_joined = polygons_joined.rename(columns={'pred_id_left': 'pred_id'})
        polygons = polygons.merge(polygons_joined[["pred_id", "k_afstand"]], on='pred_id')

    # Save the transformed GeoDataFrame
    if frommask == False: # normal prediction will make an output folder
        polygons.to_file(os.path.join(args.output, 'output', os.path.splitext(rastername)[0] + "_"+str(model_name)+"_polygons.shp"),
                            crs=crs)
    else: # shapefiles directly based on mask not
        polygons.to_file(os.path.join(args.output, os.path.splitext(rastername)[0] + "_" + str(model_name) + "_polygons.shp"),
                         crs=crs)

    print("-----------------------------------")
    print("Klaar! Zie " + os.path.splitext(rastername)[0] + "_"+str(model_name)+"_polygons.shp")
    print("-----------------------------------")

# Run
if __name__ == "__main__":
    main()