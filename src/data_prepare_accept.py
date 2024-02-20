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
import matplotlib.pyplot as plt
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

# Import helpers
# from helpers.def_custom_sliced_predict import *

from gooey import Gooey, GooeyParser # GUI

# ==============================================================

# Arguments

# ==============================================================

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Bekijk 'temp/plot.png', accepteer je het?")
def main():

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("accepteer", type=str, choices=['Ja', 'Nee'], default='Nee', widget="Dropdown", help = "Kopieer de data naar de dataset folder")
    parser.add_argument("--locatie", type=str, default=None, help="Dataset directory")
    args = parser.parse_args()

    # Write
    write_data(args)

def write_data(argus):
    import data_prepare
    print(data_prepare.args.dataset)
    
if __name__ == "__main__":
    main()