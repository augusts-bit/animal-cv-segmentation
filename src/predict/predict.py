# ==============================================================

# Setup and import

# ==============================================================

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale
# print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# YOLO and torch (check for correct versions)
import torch, detectron2
from ultralytics import YOLO

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
setup_logger()
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

# Install GDAL like this?
import subprocess
import sys

# Import helpers
from helpers.mask2shape import *

from gooey import Gooey, GooeyParser # GUI

# ==============================================================

# Arguments

# ==============================================================

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Segmenteer vogels aan de hand van drone beelden.")
def main():
    def range_limited_float_type(arg):
        """ Type function for argparse - a float within some predefined bounds """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < 0 or f > 1:
            raise argparse.ArgumentTypeError("Ingevoerde overlap of threshold is niet <= " + str(1) + "en >= " + str(0))
        return f

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("input", type=str, widget="FileChooser", # TIF file of input orthophoto
                        help="Raster dat gebruikt wordt als input")
    parser.add_argument("output", type=str, widget="DirChooser", default=os.getcwd(), # Output location
                        help="Locatie voor output (een folder wordt aangemaakt)")
    parser.add_argument("modeltype", type=str, choices=['Mask R-CNN', 'YOLOv8'], default='YOLOv8', widget="Dropdown",
                        help="Geef aan welk model")
    parser.add_argument("model", type=str, widget="DirChooser", # subfolder name of location of model
                        help="Folder locatie van model weights (*.pt, *.pth), configuratie (*.yaml) en categories.json")
    parser.add_argument("backup", type=str, choices=['Ja', 'Nee'], default='Ja', widget="Dropdown",
                        help="Wil je een backup Numpy mask maken voor predict_from_mask.py? Kan zwaar zijn bij grote rasters")
    parser.add_argument("--grootte", type=int, default=5, widget='IntegerField',
                        help="Horizontale en verticale grootte (m) van de geknipte foto's")
    parser.add_argument("--overlap", type=range_limited_float_type, default=0.5, widget='DecimalField',
                        help="Horizontale en verticale overlap fractie tussen geknipte fotos")
    parser.add_argument("--threshold", type=range_limited_float_type, default=0.3, widget='DecimalField',
                        help="Model zekerheid fractie")
    args = parser.parse_args()

    # Run Detectron2 or YOLOv8 prediction script
    if args.modeltype == "YOLOv8":
        import yolov8_predict  # YOLO prediction
        yolov8_predict.main(args)
    if args.modeltype == "Mask R-CNN":
        import detectron2_predict  # Mask R-CNN prediction
        detectron2_predict.main(args)
    else:
        print("Geen geldige modeltype")

if __name__ == "__main__":
    main()