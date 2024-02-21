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

from gooey import Gooey, GooeyParser # GUI
from formlayout import fedit # form to accept

# ==============================================================

# Arguments

# ==============================================================

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Maak een dataset voor vogelsegmentatie/classificatie.")
def main():

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("raster", type=str, widget="FileChooser",
                        help="Raster als input (zoals een .tif)")
    parser.add_argument("shapes", type=str, widget="FileChooser",
                        help="Shapes als input (.shp)")
    parser.add_argument("dataset", type=str, widget="DirChooser", default=os.getcwd(),
                        help="Folder waar de dataset is/moet komen")
    args = parser.parse_args()

    # Run data preparation
    prepare_data(args)

    ########################################
    # prompt to accept the data
    ########################################

    R = fedit(title='Data toevoegen aan dataset?',
              comment='Check of de data goed in de "temp" folder goed is (zie bv. de plots)',
              data=[('Toevoegen?', [2, 'Ja', 'Nee'])])

    if R[0] == 1: # Ja
        accept_data(args)
    if R[0] == 2: # Nee
        print("----------------------------")
        print("Je hebt de data niet toegevoegd")
        print("----------------------------")

    # Execute another script
    # PYTHON_PATH = sys.executable
    # process = Popen([PYTHON_PATH, 'data_accept.py', args.dataset], stdout=PIPE, stderr=PIPE)
    # output, error = process.communicate()
    # print(output)
    # print(error)

def prepare_data(args):
    # Load shapes
    gdf = gpd.read_file(args.shapes)

    # Create 'temp' folder to store temporary data
    if os.path.exists(os.path.join(args.dataset, "temp")):
        shutil.rmtree(os.path.join(args.dataset, "temp")) # delete old temp folder
    os.makedirs(os.path.join(args.dataset, "temp"), exist_ok=True)

    # Get soorten
    soorten = create_soorten(os.path.join(args.dataset, "soorten.json"), gdf)

    # Check if raster needs resample (and indirectly if can be loaded)
    resample, r_bool = needs_resample(args.raster)
    if r_bool:
        print("----------------------------")
        print(resample)
        print("----------------------------")

    # Rasterise shapes
    print("----------------------------")
    print("Rasteriseren van de shapes...")
    mask = rasterise(args, gdf, soorten)
    print("----------------------------")

    # Tile
    print("----------------------------")
    raster_tiles, mask_tiles = tile(args.raster, os.path.join(args.dataset, "temp", "full_mask.tif"))

    # Filter tiles
    raster_tiles, mask_tiles = shuffle_filter_tiles(raster_tiles, mask_tiles)
    print("----------------------------")

    # Plot to check
    print("----------------------------")
    check_plot(raster_tiles, mask_tiles, args)
    print("----------------------------")

    # Make data (in temp folder first)
    print("----------------------------")
    rastername = os.path.basename(os.path.splitext(args.raster)[0])
    write2data(args.dataset, rastername, raster_tiles, mask_tiles, soorten)
    print("----------------------------")

    # Data is prepared, ask user for prompt to continue
    print("----------------------------")
    print("Data is gemaakt en zit nu in de 'temp' folder")
    print("Voordat het wordt overgeschreven naar de daadwerkelijke locatie, check het eerst")
    print("Bekijk bijvoorbeeld naar de plots in de 'temp' folder of er geen ruimtelijke mismatch is")
    print("In de prompt, klik op 'Ja' als het goed is en je de data wilt toevoegen")
    print("----------------------------")

    print("----------------------------")
    print("Werkt het niet? Kopieer en plak de data dan zelf uit de temp folder")
    print("----------------------------")

def accept_data(args):
    # Images
    folder_destination = os.path.join(args.dataset, "images")
    for file_name in os.listdir(os.path.join(args.dataset, "temp", "images")):

        # construct full file path
        source = os.path.join(args.dataset, "temp", "images", file_name)
        destination = os.path.join(folder_destination, file_name)

        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

    print(source)
    print(destination)
    
    # Masks
    folder_destination = os.path.join(args.dataset, "masks")
    for file_name in os.listdir(os.path.join(args.dataset, "temp", "masks")):

        # construct full file path
        source = os.path.join(args.dataset, "temp", "masks", file_name)
        destination = os.path.join(folder_destination, file_name)

        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

    # Annotations
    folder_destination = os.path.join(args.dataset, "annotations")
    for file_name in os.listdir(os.path.join(args.dataset, "temp", "annotations")):

        # construct full file path
        source = os.path.join(args.dataset, "temp", "annotations", file_name)
        destination = os.path.join(folder_destination, file_name)

        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

    # Labels
    folder_destination = os.path.join(args.dataset, "labels")
    for file_name in os.listdir(os.path.join(args.dataset, "temp", "labels")):

        # construct full file path
        source = os.path.join(args.dataset, "temp", "labels", file_name)
        destination = os.path.join(folder_destination, file_name)

        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

    print("----------------------------")
    print("Data toegevoegd!")
    print("----------------------------")

if __name__ == "__main__":
    main()