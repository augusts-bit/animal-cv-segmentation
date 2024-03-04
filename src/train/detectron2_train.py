# Based on https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=h9tECBQCvMv3

# ==============================================================

# Setup and import

# ==============================================================

# Install detectron2

# To avoid error: NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale
print(locale.getpreferredencoding())

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# %%
import sys, os, distutils.core

# Detectron and torch (check for correct versions)
import torch, detectron2

# Parse arguments given to job
import argparse

# Setup detectron2 logger
import detectron2
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
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.data import transforms as T
from sahi.utils.detectron2 import export_cfg_as_yaml

# Other
from PIL import Image
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2
import time
import datetime
import random

# Log metrics
import mlflow
import mlflow.pytorch

# Import helpers scripts
from helpers.hooks import *
from helpers.loadDatadict import *
from helpers.getCategoriesSizes import *

# ==============================================================

# Load inputs given to job

# ==============================================================

# This is how you load input from job
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-train-model?view=azureml-api-2 

# input and output arguments
parser = argparse.ArgumentParser()

# Training data
parser.add_argument("--traindata")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.001)
parser.add_argument("--img_per_batch", type=int, default=3)
parser.add_argument("--roi_batch_size", type=int, default=16)
parser.add_argument("--rescale", type=int, default=0) # no rescale
args = parser.parse_args()

# create output directory just in case
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/model', exist_ok=True)

# ==============================================================

# Check what species and image sizes are in dataset

# ==============================================================

# Check if there is a soorten.json, if not, create soorten list but this may cause classification errors (faulty indexes)
if os.path.isfile(os.path.join(args.traindata, "soorten.json")):
    with open(os.path.join(args.traindata, "soorten.json"), 'r') as json_file:
        soorten = json.load(json_file)
    print("Dataset contains:", soorten)
else:
    soorten, min_width, min_height = get_classes_sizes(os.path.join(args.traindata, "annotations"))
    print("Warning, no soorten.json found!")
    print("Dataset now contains:", soorten)
    print("Minimum width and heigth:", min_width, "x", min_height) # maybe base rescaling of this

# save dictionary of classes
soorten_dict = {str(index): bird for index, bird in enumerate(soorten)}
with open('outputs/model_categories.json', 'w') as json_file:
    json.dump(soorten_dict, json_file)    
    
# ==============================================================

# Load a custom data set

# ==============================================================

# Number of files before dataset                    
print(
    len(os.listdir(os.path.join(args.traindata, "images"))), "images", 
    len(os.listdir(os.path.join(args.traindata, "masks"))), "masks",  
    len(os.listdir(os.path.join(args.traindata, "annotations"))), "annotations",
    "before dataset creation"
) 

# Get dataset of dicts (prints how many files in dataset)
train_dicts = get_custom_dicts(os.path.join(args.traindata, "images"),
                    os.path.join(args.traindata, "annotations"),
                    os.path.join(args.traindata, "masks"), soorten) # dataset needs to know how many and what species

print("Got dataset")

# Split dataset of dicts into training and validation sets
def split_dataset(dataset, split_ratio=0.9): # 0.8 or 0.9?
    
    # Shuffle the dataset randomly
    random.shuffle(dataset)
    
    # Split
    total_samples = len(dataset)
    split_idx = int(total_samples * split_ratio)

    return dataset[:split_idx], dataset[split_idx:], total_samples

# Get training and validation dicts
train_dicts, vali_dicts, train_length = split_dataset(train_dicts)
print("Images used for training:", int(train_length * 0.9))
print("Images used for validation:", int(train_length * 0.1))
num_iterations = int((train_length*0.9)/args.img_per_batch) * args.epochs # num iterations
print("Will do", num_iterations, "iterations")

# Clear/overwrite registered datasets (if necessary)
DatasetCatalog.clear()

# Register datasets
DatasetCatalog.register("train", lambda:train_dicts)
MetadataCatalog.get("train").set(thing_classes=soorten)
train_metadata = MetadataCatalog.get("train")

DatasetCatalog.register("vali", lambda:vali_dicts)
MetadataCatalog.get("vali").set(thing_classes=soorten)
vali_metadata = MetadataCatalog.get("vali")

# ==============================================================

# Config

# ==============================================================

# Clear cache
torch.cuda.empty_cache() 

# Base parameters
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) 
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("vali",) 
# cfg.TEST.EVAL_PERIOD = 100 # do in hook
cfg.DATALOADER.NUM_WORKERS = 2 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 

# Solver (learning rate and decay)
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.IMS_PER_BATCH = args.img_per_batch  
cfg.SOLVER.BASE_LR = args.learning_rate  
cfg.SOLVER.MAX_ITER = num_iterations   
cfg.SOLVER.GAMMA = args.gamma
cfg.SOLVER.STEPS = [2*int(num_iterations/5),
                   3*int(num_iterations/5),
                   4*int(num_iterations/5)]        # decay learning rate at iterations by gamma
cfg.SOLVER.CHECKPOINT_PERIOD = round(cfg.SOLVER.MAX_ITER/10) # save checkpoints to watch for underfitting

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(soorten)   # number of soorten is number of classes
cfg.INPUT.MASK_FORMAT = "bitmask" # Set the MASK_FORMAT to bitmask

# Necessary to train on images with no objects/birds
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.OUTPUT_DIR = "outputs/model" # checkpoints will be saved here

# Resize images
cfg.INPUT.MIN_SIZE_TRAIN = (args.rescale,) # Better to downscale than to upscale (take smallest train image), 0 for no resizing
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice" # can do multiple rescales
cfg.INPUT.MIN_SIZE_TEST = args.rescale 

# Detect more instances? (https://github.com/facebookresearch/detectron2/issues/1481) 
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000  

# Other custom parameters
# cfg.MODEL.RPN.IN_FEATURES, cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, cfg.MODEL.ANCHOR_GENERATOR.SIZES
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION, cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS

# Check if a GPU is available
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cuda' # Otherwise will get Runtime error as no NVIDIA
else:
    cfg.MODEL.DEVICE = 'cpu'

# ==============================================================

# Train

# ==============================================================         

# Create trainer with augmentations
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.RandomBrightness(0.9, 1.1), # Don't make brightness, contrast or saturation effect too high
                T.RandomContrast(0.9, 1.1),
                T.RandomSaturation(0.9, 1.1),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            ]
        )
        return build_detection_train_loader(cfg, mapper=mapper)

trainer = MyTrainer(cfg) # otherwise use DefaultTrainer

# Add hooks (for logging in Azure and to measure epoch loss)
data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg,True))
epoch_iter = int((train_length * 0.9)/cfg.SOLVER.IMS_PER_BATCH) # number of iterations per epoch
epoch_iter_checkpoint = int(epoch_iter/10)  # number of iterations per checkpoint
ml_flow_hook = MLflowHook(epoch_iter_checkpoint) # hook of loss per checkpoint or epoch?
# trainer.register_hooks([ml_flow_hook]) # Commented out, because hooks make training longer

# Train
trainer.resume_or_load(resume=False)
trainer.train()

# ==============================================================

# Save the (latest) model and its weights

# ==============================================================

# Save in job (these are the last (iteration) weigths, check losses if checkpoint model is better
torch.save(trainer.model.state_dict(), os.path.join("outputs/model", "model_weights.pt"))
export_cfg_as_yaml(cfg, export_path=os.path.join("outputs/model", "model_cfg.yaml"))
# trainer.model.save_model(cfg.MODEL.WEIGHTS) # not working (?)

# MLFLOW: mlflow has a nice method to export the model automatically
# add tags and environment for it. You can then use it in Azure ML
# to register your model to an endpoint.

# # Register as model
# mlflow.pytorch.log_model(
#     trainer.model,
#     artifact_path="outputs/model", #os.path.join("outputs/model", "model_weights.pt"),
#     registered_model_name="test-model",  # also register it if name is provided
#     # signature=self.model_signature, # perhaps better than description (?)
# )

# ==============================================================

# Visualise (log images)

# ==============================================================

# Set to cpu
cfg.MODEL.DEVICE = 'cpu'

# Inference should use the config with parameters that are used in training
cfg.MODEL.WEIGHTS = os.path.join("outputs/model", "model_weights.pt")

# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Randomly select and visualise samples of vali predictions
img_nr = 1

for d in random.sample(vali_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=vali_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. 
                    )
    out = v.draw_instance_predictions(outputs["instances"].to(cfg.MODEL.DEVICE))

    # Log image 
    mlflow.log_image(out.get_image()[:, :, ::-1], "rnd_pred_img"+str(img_nr)+".png")
    img_nr = img_nr + 1

