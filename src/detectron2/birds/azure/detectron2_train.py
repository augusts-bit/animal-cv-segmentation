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

# %%
# Some basic setup:
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
from sahi.utils.detectron2 import export_cfg_as_yaml

# Other
from PIL import Image
from torchvision.io import read_image
from pycocotools import mask as coco_mask
import cv2

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

# Training data
parser.add_argument("--traindata")
args = parser.parse_args()

# Hyper parameters
with open('param.json', 'r') as json_file:
    param = json.load(json_file)
iterations = param["iterations"]
learning_rate = param["learning_rate"]
img_per_batch = param["img_per_batch"]
roi_batch_size = param["roi_batch_size"]

# ==============================================================

# Load a custom data set

# ==============================================================

# See what needs to be in a dataset:
# https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# Custom function if not already in COCO format (format to own)
def get_custom_dicts(img_dir, ann_dir, mask_dir):

    dataset_dicts = []

    for idx, ann_file in enumerate(sorted(os.listdir(ann_dir))):

        record = {}

        # Get image path
        imgs = list(sorted(os.listdir(img_dir)))
        img_path = os.path.join(img_dir, imgs[idx])

        # Load image
        im = Image.open(img_path)
        width, height = im.size

        # Get segmentation/masks
        masks = list(sorted(os.listdir(mask_dir)))
        mask_path = os.path.join(mask_dir, masks[idx])

        mask = read_image(mask_path)
        mask = mask[0].cpu().numpy()
        binary_mask = (mask > 0) # important

        rle_dict = coco_mask.encode(np.asarray(binary_mask, order="F"))

        # Add basic image information to record
        record["file_name"] = img_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Save annotations in a list (remains empty if no annotations - only background)
        objs = []

        # Get annotations: bounding box, segmentation (mask) and label
        with open(os.path.join(ann_dir, ann_file), mode="r") as file:

            annotations = file.readlines()

            bboxes = []

            # Get bounding box
            for annotation in annotations:
                
                # Process each line in the file
                if annotation.startswith('Bounding box'):

                    # Extract bounding box coordinates
                    match = re.search(r'\((\d+), (\d+)\) - \((\d+), (\d+)\)', annotation)
                    if match:
                        x, y, x_max, y_max = map(int, match.groups())
                        bbox = [x, y, x_max, y_max]
                        bboxes.append(bbox)

            if bbox: # (there are objects in the image)
            # if len(bboxes) > 0: 

                for j in range(len(bboxes)):
                    x, y, x_max, y_max = bboxes[j]
                    w = x_max - x
                    h = y_max - y

                    # Create the object dictionary
                    obj = {
                        "bbox": [x, y, x + w, y + h],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": rle_dict,
                        "category_id": 0,  # If only one class, id = 0
                    }

                    # Save annotation to annotations list
                    objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts

# Get dataset of dicts
dataset_dicts = get_custom_dicts(os.path.join(args.traindata, "Images"),
                    os.path.join(args.traindata, "Annotations"),
                    os.path.join(args.traindata, "Masks"))

print("Got dataset")

# Split dataset of dicts into training and validation sets
def split_dataset(dataset, split_ratio=0.9): # 0.8 or 0.9?
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (list): The dataset to be split.
        split_ratio (float): The ratio of the dataset to be used for training.

    Returns:
        tuple: Two lists, first is training set, second is validation set.
    """
    total_samples = len(dataset)
    split_idx = int(total_samples * split_ratio)

    return dataset[:split_idx], dataset[split_idx:]

# Get training and validation dicts
train_dicts, vali_dicts = split_dataset(dataset_dicts)

# Clear/overwrite registered datasets (if necessary)
DatasetCatalog.clear()

# Register datasets
DatasetCatalog.register("all", lambda:dataset_dicts)
MetadataCatalog.get("train").set(thing_classes=["Bird"])
dataset_metadata = MetadataCatalog.get("all")

DatasetCatalog.register("train", lambda:train_dicts)
MetadataCatalog.get("train").set(thing_classes=["Bird"])
train_metadata = MetadataCatalog.get("train")

DatasetCatalog.register("vali", lambda:vali_dicts)
MetadataCatalog.get("vali").set(thing_classes=["Bird"])
vali_metadata = MetadataCatalog.get("vali")

# Check if correct
# img_nr = 1

# for d in random.sample(train_dicts, 5):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1.5)

#     out = visualizer.draw_dataset_dict(d)
#     # cv2_imshow(out.get_image()[:, :, ::-1])

#     # Log image 
#     mlflow.log_image(out.get_image()[:, :, ::-1], "rnd_train_img"+str(img_nr)+".png")
#     img_nr = img_nr + 1

# ==============================================================

# Config

# ==============================================================

#  Parameters

# "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" used by https://www.mdpi.com/2072-4292/12/18/3015 (crops segmentation)?

# Base parameters
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # Used "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" before
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("vali",) # Or should use different one here ?
# cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 2 # was 2
# cfg.DATALOADER.PREFETCH_FACTOR = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo (used R 50 FPN before)

cfg.SOLVER.IMS_PER_BATCH = img_per_batch  # Was 2. This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = learning_rate  # (was 0.00025) pick a good LR
cfg.SOLVER.MAX_ITER = iterations   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size # was 16  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 if only one class. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# Necessary to train on images with no objects/birds (?)
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

cfg.OUTPUT_DIR = "Model"

# Custom parameters
# Adjust the parameters for detecting smaller objects (?)

# cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0]] #, 2.0, 4.0, 8.0]]
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8], [16], [32], [64], [128]]
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 10240
# cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
# cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5] # Intersection over union threshold

# Set the MASK_FORMAT to bitmask
cfg.INPUT.MASK_FORMAT = "bitmask" # important (?)

# Check if a GPU is available
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cuda' # Otherwise will get Runtime error as no NVIDIA
    # cfg.MODEL.module.to("cuda")
else:
    cfg.MODEL.DEVICE = 'cpu'
    # cfg.MODEL.module.to("cpu")

# ==============================================================

# Train and log (only doing that for Azure)

# ==============================================================

# Create loss classes to log the losses (losses don't fully match the metrics given by default?)
class TrainingLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = self.cfg.DATASETS.TRAIN
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)
            
            # print(f"Training Loss (Iteration {self.trainer.iter}): {losses_reduced}")
            mlflow.log_metric('train_loss',losses_reduced)

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)
            
            # print(f"Vali Loss (Iteration {self.trainer.iter}): {losses_reduced}")
            mlflow.log_metric('vali_loss',losses_reduced)

# How you log in Azure
# mlflow.autolog()
# mlflow.log_metric('anothermetric',1)

# Create trainer and train
trainer = DefaultTrainer(cfg)
val_loss = ValidationLoss(cfg)
train_loss = TrainingLoss(cfg)
trainer.register_hooks([val_loss])
trainer.register_hooks([train_loss])
trainer.resume_or_load(resume=False)
trainer.train() # Train!

# # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# cfg.DATALOADER.PREFETCH_FACTOR = 2
# trainer = DefaultTrainer(cfg)
# val_loss = ValidationLoss(cfg)  
# trainer.register_hooks([val_loss])
# # swap the order of PeriodicWriter and ValidationLoss
# trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
# # trainer = CustomTrainer(cfg) # Overwrite default trainer
# trainer.resume_or_load(resume=False)
# trainer.train() # Train!

# ==============================================================

# Save the model and its weights

# ==============================================================

# create output directory just in case
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/model', exist_ok=True)

# Save in job
torch.save(trainer.model.state_dict(), os.path.join("outputs/model", "model_weights.pt"))
export_cfg_as_yaml(cfg, export_path=os.path.join("outputs/model", "model_cfg.yaml"))
# trainer.model.save_model(cfg.MODEL.WEIGHTS) # not working (?)

# MLFLOW: mlflow has a nice method to export the model automatically
# add tags and environment for it. You can then use it in Azure ML
# to register your model to an endpoint.

# Read the contents of param.json
with open('param.json', 'r') as json_file:
    description = json.load(json_file)

# Give as description to the model
with open(os.path.join("outputs/model", "param.json"), 'w') as f:
    json.dump(description, f)

# Register as model
mlflow.pytorch.log_model(
    trainer.model,
    artifact_path="outputs/model", #os.path.join("outputs/model", "model_weights.pt"),
    registered_model_name="test-model",  # also register it if name is provided
    # signature=self.model_signature, # perhaps better than description (?)
)

# ==============================================================

# Visualise (log images)

# ==============================================================

# Inference should use the config with parameters that are used in training
cfg.MODEL.WEIGHTS = os.path.join("outputs/model", "model_weights.pt")

# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Randomly select and visualise samples of vali predictions

img_nr = 1

for d in random.sample(vali_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=vali_metadata,
                   scale=1.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only    available for segmentation models
                    )
    out = v.draw_instance_predictions(outputs["instances"].to(cfg.MODEL.DEVICE))
    # cv2.cv2_imshow(out.get_image()[:, :, ::-1])

    # Log image 
    mlflow.log_image(out.get_image()[:, :, ::-1], "rnd_pred_img"+str(img_nr)+".png")
    img_nr = img_nr + 1

