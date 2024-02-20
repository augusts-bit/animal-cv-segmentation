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

@Gooey(program_name="Vogel Segmentatie", language="dutch", program_description="Maak een dataset voor vogelsegmentatie/classificatie.")
def main():

    # Argument parser
    parser = GooeyParser()
    parser.add_argument("raster", type=str, widget="FileChooser",
                        help="Raster dat gebruikt wordt als input")
    parser.add_argument("shapes", type=str, widget="FileChooser",
                        help="Shapes als input")
    parser.add_argument("dataset", type=str, widget="DirChooser", default=os.getcwd(),
                        help="Folder waar de dataset is/moet komen")
    args = parser.parse_args()

    # Run data preparation
    prepare_data(args)

    ########################################
    # create another Gooey where users are prompted to accept the data
    ########################################

    PYTHON_PATH = sys.executable
    process = Popen([PYTHON_PATH, "data_prepare_accept.py" "--locatie", args.dataset], stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()
    print(output)
    print(error)

# Obtain list of soorten, check if there are already (other) soorten in dataset
def create_soorten(path, shapes):

    # Check if already a data set with species
    if os.path.isfile(path):
        with open(path, 'r') as json_file:
            orig_soorten = json.load(json_file)
            new_soorten = orig_soorten + shapes["soort"].unique().tolist() # append list with new soorten
            soorten = [] # remove duplicate soorten while keeping the index
            for soort in new_soorten:
                if soort not in soorten:
                    soorten.append(soort)
        # Write new soorten list to file
        with open(path, 'w') as f:
            json.dump(soorten, f)
    else:
        soorten = shapes["soort"].unique().tolist()
        # Write to file
        with open(path, 'w') as f:
            json.dump(soorten, f)

    return soorten

# Check if raster needs resample
def needs_resample(path):
    with rasterio.open(path) as src:

        if (src.res[0] / src.res[1]) > 0.99 and (src.res[0] / src.res[1]) < 1.01:
            resample = "X en Y zijn (ongeveer) hetzelfde dus je hoeft de raster niet te resamplen"
            r_bool = False
        else:
            resample = "Waarschuwing: X en Y zijn niet gelijk en je moet de raster misschien resamplen"
            r_bool = True

        return resample, r_bool

def rasterise(args, shapes, soorten):
    with rasterio.open(args.raster) as src:
        # Generate masks based on the species
        n = 1  # Give species raster values starting from 1

        soort_raster = np.zeros_like(src.read(1))

        # Change gdf crs to that of the raster
        target_crs = src.crs
        gdf = shapes.to_crs(target_crs)  # probably better to directly do so using ArcGIS Pro

        for soort in soorten:

            # Check if soort exists within the shapes
            if soort in gdf['soort'].values:

                # Generate mask
                mask = geometry_mask(gdf[gdf['soort'] == soort].geometry, out_shape=src.shape, transform=src.transform,
                                     invert=True)

                # Set value
                soort_raster[mask] = n
                print(soort, "gevonden, wordt waarde", n)
            else:
                pass  # raster is not updated as soort was not found

            n = n + 1
        print("De achtergrond is waarde 0")

        # Save
        profile = src.profile  # use dst for reprojected
        profile.update(count=1, dtype='uint8', nodata=None)  # Remove nodata value
        with rasterio.open(os.path.join(args.dataset, "temp", "full_mask.tif"), 'w', **profile) as newr:
            newr.write(soort_raster.astype('uint8'), 1)

    return soort_raster

# Tile
def tile(imgpath, maskpath):
    # Open with GDAL
    raster_data_set = gdal.Open(imgpath)
    mask_data_set = gdal.Open(maskpath)

    # Get cell size
    gt = raster_data_set.GetGeoTransform()
    pixelSizeX = gt[1]

    # Convert to array image
    raster_band_1 = raster_data_set.GetRasterBand(1)  # red channel
    raster_band_2 = raster_data_set.GetRasterBand(2)  # green channel
    raster_band_3 = raster_data_set.GetRasterBand(3)  # blue channel
    mask_band_1 = mask_data_set.GetRasterBand(1)  # first/only channel

    # Convert to array image
    raster_b1 = raster_band_1.ReadAsArray()
    raster_b2 = raster_band_2.ReadAsArray()
    raster_b3 = raster_band_3.ReadAsArray()
    mask_b1 = mask_band_1.ReadAsArray()

    # Stack to image
    raster_img = np.dstack((raster_b1, raster_b2, raster_b3))
    mask_img = np.dstack((mask_b1))
    mask_img = mask_img[0].T  # Transpose because axis was inverted (because of invert?)

    # cell size of Texel PHZ
    target_cs = 0.011272670412636411

    # WxH tile pixel size (depending on cell size)
    W = round(890 * (target_cs / pixelSizeX))  # ~ 10.0 m x 10.0 m # was 1000
    H = round(445 * (target_cs / pixelSizeX))  # ~ 5.0 m x 5.0 m # was 350

    overlap_percentage = 0.5  # make extra tiles with overlap (every bird will be accounted this way, not cut in half)

    # Calculate the overlap size in pixels
    overlap_size_W = int(W * overlap_percentage)
    overlap_size_H = int(H * overlap_percentage)

    # Function to create tiles with overlap
    def create_overlapping_tiles(img, size_x, overlap_size_x):
        tiles = []
        step_x = size_x - overlap_size_x
        step_y = size_x - overlap_size_x

        for x in range(0, img.shape[0] - size_x + 1, step_x):
            for y in range(0, img.shape[1] - size_x + 1, step_y):
                tile = img[x:x + size_x, y:y + size_x]
                tiles.append(tile)

        return tiles

    # Split in to M and N tiles
    raster_tiles_W = [raster_img[x:x + W, y:y + W] for x in range(0, raster_img.shape[0], W) for y in
                      range(0, raster_img.shape[1], W)]
    mask_tiles_W = [mask_img[x:x + W, y:y + W] for x in range(0, mask_img.shape[0], W) for y in
                    range(0, mask_img.shape[1], W)]
    print(len(raster_tiles_W), "tiles gemaakt met aantal (horizontale en verticale) pixels:", W)

    return raster_tiles_W, mask_tiles_W

def shuffle_filter_tiles(raster_tiles, mask_tiles, achtergrond = 0):
    def shuffle_similarly(list1, list2):
        combined = list(zip(list1, list2))
        random.shuffle(combined)
        shuffled_list1, shuffled_list2 = zip(*combined)
        return list(shuffled_list1), list(shuffled_list2)

    def filter_tiles(raster_tiles, mask_tiles, bg_keep_pc):
        # Filter based on tiles where there are no masks (birds)
        bird_raster_tiles = []
        bird_mask_tiles = []
        no_bird_raster_tiles = []
        no_bird_mask_tiles = []

        for i in range(len(raster_tiles)):

            mask_tile_img = mask_tiles[i].copy()
            contours, _ = cv2.findContours(mask_tile_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours == ():
                no_bird_raster_tiles.append(raster_tiles[i])
                no_bird_mask_tiles.append(mask_tiles[i])

            else:
                bird_raster_tiles.append(raster_tiles[i])
                bird_mask_tiles.append(mask_tiles[i])

        # Filter a percentage of the no bird tiles (to reduce dataset)
        remove_count = len(no_bird_raster_tiles) * (100 - bg_keep_pc) // 100
        indices_to_remove = random.sample(range(len(no_bird_raster_tiles)), remove_count)
        no_bird_raster_tiles = [item for i, item in enumerate(no_bird_raster_tiles) if i not in indices_to_remove]
        no_bird_mask_tiles = [item for i, item in enumerate(no_bird_mask_tiles) if i not in indices_to_remove]

        # Concat and shuffle
        raster_tiles = bird_raster_tiles + no_bird_raster_tiles
        mask_tiles = bird_mask_tiles + no_bird_mask_tiles
        raster_tiles, mask_tiles = shuffle_similarly(raster_tiles, mask_tiles)
        return raster_tiles, mask_tiles

    # Apply
    raster_tiles_W0, mask_tiles_W0 = filter_tiles(raster_tiles, mask_tiles, achtergrond)
    print("Na filteren:", len(raster_tiles_W0), "tiles")
    return raster_tiles_W0, mask_tiles_W0

# Check with plot
def check_plot(raster_tiles, mask_tiles, args):
    # Create a figure with two subplots arranged in a 1x2 grid
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # See random tile
    tile_numb = random.randint(0, len(raster_tiles) - 1)

    # Plot the images
    axs[0].imshow(raster_tiles[tile_numb].astype(np.uint8))
    axs[0].axis('off')  # Turn off axis labels and ticks
    axs[1].imshow(mask_tiles[tile_numb].astype(np.uint8))
    axs[1].axis('off')

    # Overlay the mask on the raster
    axs[2].imshow(raster_tiles[tile_numb].astype(np.uint8), cmap='viridis')  # Change cmap if needed
    axs[2].imshow(mask_tiles[tile_numb].astype(np.uint8), alpha=0.5, cmap='gray')  # Adjust alpha as needed
    axs[2].axis('off')

    # Save the figure
    fig.savefig(os.path.join(args.dataset, "temp", "plot.png"))

    # Close the figure to avoid memory issues
    plt.close(fig)

# Write data
def write2data(root_path, rastername, raster_tiles, mask_tiles, soorten):

    # Create folders if they do not yet exist
    os.makedirs(os.path.join(root_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "temp", "images"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "temp", "masks"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "temp", "annotations"), exist_ok=True)

    # Write first to temp folder
    out_img_path = os.path.join(root_path, "temp", "images")
    out_mask_path = os.path.join(root_path, "temp", "masks")
    out_anno_path = os.path.join(root_path, "temp", "annotations")

    for i in range(len(raster_tiles)):

        # Output path
        out_img = os.path.join(out_img_path, rastername + "_img" + str(i + 1) + ".png")
        out_mask = os.path.join(out_mask_path, rastername + "_mask" + str(i + 1) + ".png")
        out_ann = os.path.join(out_anno_path, rastername + "_annotation" + str(i + 1) + ".txt")

        # Save images to file
        plt.imsave(out_img, raster_tiles[i].astype(np.uint8), cmap='gray', format='png')  # Convert to uint8
        plt.imsave(out_mask, mask_tiles[i].astype(np.uint8), cmap='gray', format='png')

        # Calculate boxes from mask
        mask_tile_img = mask_tiles[i].copy()
        contours, _ = cv2.findContours(mask_tile_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bird_labels = []  # List to store bird labels corresponding to each contour

        for contour in contours:
            # Find the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region of interest (ROI)
            roi = mask_tile_img[y:y + h, x:x + w]

            # Find the unique values and their counts in the ROI
            unique_values, counts = np.unique(roi, return_counts=True)

            # Exclude 0 from consideration (is background)
            counts = counts[unique_values != 0]
            unique_values = unique_values[unique_values != 0]

            # Get the index of the maximum count (majority value) --> necessary in cases of overlap
            majority_index = np.argmax(counts)

            # Assign the majority value
            bird_label = unique_values[majority_index]
            bird_labels.append(bird_label)

        # Image width and height
        # img_data, metadata = imread(out_img)
        width, height = 50, 50 # metadata['width'], metadata['height']

        # img_colour = Image.open(out_img)  # You need to have the colour image somehow (so write image to file first)
        # width, height = img_colour.size

        # Class and number of objects
        num_objects = len(contours)

        with open(out_ann, 'w') as f:
            # Write annotation to file
            f.write(f'Image filename : "{out_img}"\n')
            f.write(f'Image size (X x Y) : {width} x {height}\n')
            f.write(f'Database : "BirdsBreeding"\n')
            f.write(f'Objects with ground truth : {num_objects} \n\n')

            # Loop over each bird (if any) and label
            for j, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                # Bird species stored in "soorten", assign species based on value
                for z, soort in enumerate(soorten):
                    if bird_labels[
                        j] == z + 1:  # Python indexing starts at 0, whereas we assigned raster values from 1 (0 is no bird)
                        class_name = soort
                    else:
                        pass

                f.write(f'# Details for object {j + 1} \n')
                f.write(f'Original label for object {j + 1} : "{class_name}" \n')
                f.write(
                    f'Bounding box for object {j + 1} : (Xmin, Ymin) - (Xmax, Ymax) : ({x_min}, {y_min}) - ({x_max}, {y_max})\n')
                f.write(f'Pixel mask for object {j + 1} : "{out_mask}"\n\n')

def prepare_data(args):
    # Load shapes
    gdf = gpd.read_file(args.shapes)

    # Create 'temp' folder to store temporary data
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
    raster_tiles, mask_tiles = tile(args.raster, os.path.join(args.dataset, "temp", "full_mask.tif"))

    # Filter tiles
    raster_tiles, mask_tiles = shuffle_filter_tiles(raster_tiles, mask_tiles)

    # Plot to check
    check_plot(raster_tiles, mask_tiles, args)

    # Make data (in temp folder first)
    rastername = os.path.basename(os.path.splitext(args.raster)[0])
    write2data(args.dataset, rastername, raster_tiles, mask_tiles, soorten)
    
if __name__ == "__main__":
    main()