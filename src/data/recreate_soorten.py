# ==============================================================

# Setup and import

# ==============================================================

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

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
import random

# Other
import sys, os, distutils.core
from torchvision.io import read_image
import cv2
from osgeo import gdal
import geopandas as gpd
from pyproj import CRS
import matplotlib.image as mpimg
from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.affinity import scale
import random
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.transform import Affine
from rasterio.warp import Resampling
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
from osgeo import gdal

# ..
import subprocess
from subprocess import Popen, PIPE
import sys

# Import functions
from helpers.funcs import *
from helpers.recreate import *

from gooey import Gooey, GooeyParser # GUI
from formlayout import fedit # form to accept

# ==============================================================

# Arguments

# ==============================================================

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Recreëer soorten.json, nieuwe labels worden aangemaakt")
def main():

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("dataset", type=str, widget="DirChooser", default=os.getcwd(),
                        help="Folder waar de dataset is")
    args = parser.parse_args()

    # Recreate soorten
    soorten = recreate_soorten(args)

    # Recreate labels
    recreate_labels(soorten, args)

def recreate_soorten(args):

    # Obtain soorten from annotations
    soorten, _, _ = get_classes_sizes(os.path.join(args.dataset, "annotations"))
    print("----------------------------")
    print("De volgende soorten zijn ontdekt en worden gebruikt:", soorten)

    # Save as json
    with open(os.path.join(args.dataset, "soorten.json"), 'w') as f:
        json.dump(soorten, f)

    print("Een nieuwe 'soorten.json' is aangemaakt")
    print("----------------------------")

    return soorten

def recreate_labels(soorten, args):

    print("----------------------------")
    print("Nieuwe labels worden aangemaakt...")

    # Rename old labels folder (if exist) first
    if os.path.isdir(os.path.join(args.dataset, "labels")):

        # First make sure to remove an old "old_labels" folder first
        if os.path.isdir(os.path.join(args.dataset, "old_labels")):
            shutil.rmtree(os.path.join(args.dataset, "old_labels"))

        # Rename and create labels folder
        os.rename(os.path.join(args.dataset, "labels"), os.path.join(args.dataset, "old_labels"))
        os.makedirs(os.path.join(args.dataset, "labels"), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.dataset, "labels"), exist_ok=True)

    # Create labels in labels folder
    make_labels(os.path.join(args.dataset, "annotations"), os.path.join(args.dataset, "masks"), os.path.join(args.dataset, "labels"), soorten)

    print("Klaar! Oude labels vind je in 'old_labels', de nieuwe in 'labels')
    print("----------------------------")

if __name__ == "__main__":
    main()