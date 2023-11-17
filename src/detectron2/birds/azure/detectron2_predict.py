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

# Parse arguments given to job
import argparse

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import re
import torch
import tensorboard
import tabulate

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
from IPython.display import Image

# Other
import sys, os, distutils.core
from PIL import Image
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2
from osgeo import gdal
import geopandas as gpd
import matplotlib.pyplot as plt
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
import mlflow
import mlflow.pytorch

# ==============================================================

# Load inputs given to job

# ==============================================================

# This is how you load input from job
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-train-model?view=azureml-api-2 

# input and output arguments
parser = argparse.ArgumentParser()

# Rasters
parser.add_argument("--rasterdata")
# parser.add_argument("--rasternames")

# Create subset?
# parser.add_argument("--create_subset")
args = parser.parse_args()

# ==============================================================

# Convert rasters to images, use subsets only if needed

# ==============================================================

# Convert dictionary to list
# rnames = list(args.rasternames.values())

# List tif files and their names
rnames = [os.path.splitext(f)[0] for f in os.listdir(args.rasterdata) if f.lower().endswith(('.tif', '.tiff'))]

# Rasters
test_folder = args.rasterdata # Folder containing rasters for prediction
rasters = rnames # E.g., ['steenplaat_vlucht1', 'waterdunen_vlucht2']
create_subset = "no"

# Create subsets if wanted
if create_subset == "yes":

    for raster in rasters:

        # Read path
        raster_data_path = os.path.join(test_folder, raster + ".tif")
        out_path = raster + "_subset" + ".tif"

        # Open the original raster file
        with rasterio.open(raster_data_path) as src:
            # Get the dimensions of the original raster
            width = src.width
            height = src.height

            # Calculate the bounding box coordinates (check what gives nice subset)
            xmin = width // 10
            ymin = height // 5
            xmax = width // 2
            ymax = height // 2

            print(xmin, ymin, xmax, ymax)

            # Define a window using these coordinates
            window = Window(xmin, ymin, xmax - xmin, ymax - ymin)

            # Read the data within the window
            quarter_data = src.read(window=window)

            # Update the transform to reflect the new window
            new_transform = src.window_transform(window)

            # Create a new raster file with the quarter data
            with rasterio.open(out_path, 'w', driver='GTiff',
                            width=xmax - xmin, height=ymax - ymin, count=src.count,
                            dtype=src.dtypes[0], crs=src.crs, transform=new_transform) as dst:
                dst.write(quarter_data)

if create_subset != "yes" and create_subset != "no":
    sys.exit("Please specify if you want to create subsets ('yes' or 'no').")

else:
    pass


# Convert tif to img
out_paths = []

for raster in rasters:

    if create_subset == "yes":

        raster_data_set = gdal.Open(raster + "_subset" + ".tif")
        out_img = raster + "_subset" + ".png"
        out_paths.append(out_img)

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

        out_img = raster + ".png"
        out_paths.append(out_img)

    # Read
    raster_band_1 = raster_data_set.GetRasterBand(1) # red channel
    raster_band_2 = raster_data_set.GetRasterBand(2) # green channel
    raster_band_3 = raster_data_set.GetRasterBand(3) # blue channel

    # Convert to array image
    raster_b1 = raster_band_1.ReadAsArray()
    raster_b2 = raster_band_2.ReadAsArray()
    raster_b3 = raster_band_3.ReadAsArray()

    # Stack to image
    raster_img = np.dstack((raster_b1, raster_b2, raster_b3))

    # Save image to file
    plt.imsave(out_img, raster_img.astype(np.uint8), cmap='gray', format='png') # Convert to uint8


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
if torch.cuda.is_available():
    device_name = 'cuda:0' 
else:
    device_name = 'cpu'

# Load detection model from trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path="model/model_weights.pt", 
    config_path="model/model_cfg.yaml", 
    confidence_threshold=0.3,
    image_size=640, # (does this matter?)
    device=device_name #'cuda:0' # "cpu"
)

# Make prediction

predict_path = r"Prediction/" # Path to store prediction

# Loop through specified rasters (now images) to make predictions
results = []

for raster, out_img in zip(rasters, out_paths):

  # Takes very long if large image
  result = get_sliced_prediction(
      out_img, # path to image
      detection_model,
      slice_height = 250, # test what suffices (assumes photos are taken with same height -- make height dependent variable ?)
      slice_width = 250, # i.d.
      overlap_height_ratio = 0.4, # The higher the overlap, the less it misses (?)
      overlap_width_ratio = 0.4,
  )

  results.append(result)

# Visualise predictions
for result, raster in zip(results, rasters):

    result.export_visuals(export_dir=predict_path, hide_labels=True)

    # Rename exported result (standard name = "prediction_visual.png")
    old_file = os.path.join(predict_path, "prediction_visual.png")
    new_file = os.path.join(predict_path, raster + "_visual.png")
    os.rename(old_file, new_file)

    # Also log
    mlflow.log_image(cv2.imread(new_file), raster + "_pred"+".png")

# ==============================================================

# Convert to shapefiles and post-process

# ============================================================== 

for result in results:
  object_prediction_list = result.object_prediction_list
  object_prediction_list[0]

# To coco annotations

coco_annos = []
for result in results:
    coco_annotations = result.to_coco_annotations()
    coco_annos.append(coco_annotations)

# Geotransform pixel locations to geographic locations

gts = []

for raster in rasters:

    if create_subset == "yes":

        ds = gdal.Open(os.path.join(test_folder, raster + "_subset" + ".tif"))
        gt = ds.GetGeoTransform()  # Get geotransform information

        # gt will contain information like pixel size, rotation, and top-left corner coordinates
        print(f"GeoTransform: {gt}")

        gts.append(gt)
    
    else:

        ds = gdal.Open(os.path.join(test_folder, raster + ".tif"))
        gt = ds.GetGeoTransform()  # Get geotransform information

        # gt will contain information like pixel size, rotation, and top-left corner coordinates
        print(f"GeoTransform: {gt}")

        gts.append(gt)

# Assuming you have the geotransform information stored in 'gt'
# gt = [top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height]

gdfs = []

for gt, coco_anno in zip(gts, coco_annos):

    # Initialize a list to store the converted polygons
    geographic_polygons = []

    # Loop through the COCO annotations
    for annotation in coco_anno:
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

    gdfs.append(gdf)

# Post-process (?)

for gdf, raster in zip(gdfs, rasters):

    colors = [random.choice(['red', 'green', 'blue', 'purple', 'orange']) for _ in range(len(gdf))]

    gdf.plot(column='geometry', color=colors, figsize=(20, 8))

    # Save the plot to an image file
    plt.savefig("before_pp_" + raster + ".png")

    # Log
    mlflow.log_image(cv2.imread("before_pp_" + raster + ".png"), "before_pp_" + raster + ".png")

    # plt.show()

# Post-process shapes based on geometry

gdf_pps = []
n = 1

for gdf in gdfs:

    print("Prediction:", n)
    n = n+1

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
    gdf_pps.append(gdf_pp)

    print("After dropping shapes that are too small:", len(gdf_pp))

    print("\n")

for gdf_pp, raster in zip(gdf_pps, rasters):

    gdf_pp.plot(column='geometry', color=colors, figsize=(20, 8))

    # Save the plot to an image file
    plt.savefig("pp_" + raster + ".png")
    
    # plt.show()

    # Log
    mlflow.log_image(cv2.imread("pp_" + raster + ".png"), "pp_" + raster + ".png")

# ==============================================================

# Save

# ============================================================== 

os.makedirs('outputs', exist_ok=True)

for gdf_pp, raster in zip(gdf_pps, rasters):

    # Define the coordinate reference system (CRS)
    # Open the raster file to get CRS

    if create_subset == "yes":
        with rasterio.open(raster + "_subset" + ".tif") as src:
            raster_crs = src.crs
    else:
        with rasterio.open(os.path.join(test_folder, raster + ".tif")) as src:
            raster_crs = src.crs

    gdf_pp.crs = raster_crs

    gdf_pp.to_file(os.path.join("outputs", raster + "_polygons.shp"))


