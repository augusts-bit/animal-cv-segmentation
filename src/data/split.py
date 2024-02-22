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
from pathlib import *

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

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Maak training en validatie .txt (voor YOLOv8)")
def main():

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("dataset", type=str, widget="DirChooser", default=os.getcwd(),
                        help="Folder waar de dataset is")
    parser.add_argument("--split", type=float, default=0.90, widget='DecimalField',
                        help="Fractie (max 0.99) gebruikt voor training, de rest (1-fractie) is validatie")
    args = parser.parse_args()

    if args.split > 0.99:
        args.split = 0.99

    # Split
    make_split(args)

def make_split(args):
    # See https://github.com/ultralytics/yolov5/issues/1579

    root_path = args.dataset
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

    def img2label_paths(img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def autosplit(path, weights, annotated_only=False):
        """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
        Usage: from utils.dataloaders import *; autosplit()
        Arguments
            path:            Path to images directory
                            --> assumes that the corresponding labels is in the same parent directory /labels/
            weights:         Train, val, test weights (list, tuple)
            annotated_only:  Only use images with an annotated txt file
        """
        path = Path(path)  # images dir
        files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
        n = len(files)  # number of files
        random.seed(0)  # for reproducibility
        indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

        txt = ['autosplit_train.txt', 'autosplit_val.txt', ]  # 2 txt files
        for x in txt:
            if (path.parent / x).exists():
                (path.parent / x).unlink()  # remove existing

        # print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
        for i, img in zip(indices, files):
            if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
                with open(path.parent / txt[i], 'a') as f:
                    f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file

    print("----------------------------")
    print("De dataset wordt gesplit met een ratio",  args.split, "(train) en", round(1-args.split, 2), "(validatie)...")
    autosplit(os.path.join(args.dataset, "images"), weights=(args.split, round(1-args.split, 2), 0.0))
    print("Klaar! Zie 'autosplit_train.txt' en 'autopslit_val.txt' in de dataset folder")
    print("----------------------------")

    if not os.path.isfile(os.path.join(args.dataset, "autosplit_val.txt")):
        print("----------------------------")
        print("Waarschuwing:")
        print("Er is geen autosplit_val.txt gemaakt")
        print("--> maak een nieuwe split met meer data of een lagere training split")
        print("--> of maak een lege autosplit_val.txt: dit kan mogelijk een probleem zijn met training in Azure (lege tekst bestanden worden niet gezien als BLOB)")
        print("----------------------------")

if __name__ == "__main__":
    main()