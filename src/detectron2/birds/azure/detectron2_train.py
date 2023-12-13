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
parser.add_argument("--iterations", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--img_per_batch", type=int)
parser.add_argument("--roi_batch_size", type=int)
args = parser.parse_args()

# ==============================================================

# Check what species are in dataset

# ==============================================================

ann_dir = os.path.join(args.traindata, "Annotations")
ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.txt')])

categories = []
for ann_file in ann_files:

       # Get annotations: bounding box, segmentation (mask) and label
        with open(os.path.join(ann_dir, ann_file), "r") as file:
            annotations = file.readlines()

            # Get category
            for annotation in annotations:
                # Process each line in the file
                if annotation.startswith('Original label'):

                    # Extract category
                    match = re.search(r'Original label for object \d+ : "(.*?)"', annotation)
                    if match:
                        label = match.group(1)
                        categories.append(label)

soorten = sorted(list(set(categories)))
print("Dataset contains:", soorten)

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
        with open(os.path.join(ann_dir, ann_file), "r") as file:
            annotations = file.readlines()

            bboxes = []
            categories = []

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

            # Get category
            for annotation in annotations:
                # Process each line in the file
                if annotation.startswith('Original label'):

                    # Extract category
                    match = re.search(r'Original label for object \d+ : "(.*?)"', annotation)
                    if match:
                        label = match.group(1)
                        categories.append(label)

            if bbox: # (there are objects in the image)

                for j in range(len(bboxes)):
                    x, y, x_max, y_max = bboxes[j]
                    w = x_max - x
                    h = y_max - y

                    # Assign category from 'soorten' list
                    for z, soort in enumerate(soorten):
                        if categories[j] == soort:
                            category = z

                    # Create the object dictionary
                    obj = {
                        "bbox": [x, y, x + w, y + h],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": rle_dict,
                        "category_id": category, # Between 0, num_classes - 1
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

    return dataset[:split_idx], dataset[split_idx:], total_samples

# Get training and validation dicts
train_dicts, vali_dicts, _ = split_dataset(dataset_dicts)
_, _, train_length = split_dataset(dataset_dicts)
print("Images used for training:", int(train_length * 0.9))
print("Images used for validation:", int(train_length * 0.1))

# Clear/overwrite registered datasets (if necessary)
DatasetCatalog.clear()

# Register datasets
DatasetCatalog.register("all", lambda:dataset_dicts)
MetadataCatalog.get("train").set(thing_classes=soorten) # classes are stored in 'soorten' list
dataset_metadata = MetadataCatalog.get("all")

DatasetCatalog.register("train", lambda:train_dicts)
MetadataCatalog.get("train").set(thing_classes=soorten)
train_metadata = MetadataCatalog.get("train")

DatasetCatalog.register("vali", lambda:vali_dicts)
MetadataCatalog.get("vali").set(thing_classes=soorten)
vali_metadata = MetadataCatalog.get("vali")

# ==============================================================

# Config

# ==============================================================

# create output directory just in case
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/model', exist_ok=True)

#  Parameters

# Base parameters
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) 
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("vali",) 
# cfg.TEST.EVAL_PERIOD = 100 # do in hook
cfg.DATALOADER.NUM_WORKERS = 2 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 

cfg.SOLVER.IMS_PER_BATCH = args.img_per_batch  
cfg.SOLVER.BASE_LR = args.learning_rate  
cfg.SOLVER.MAX_ITER = args.iterations   
cfg.SOLVER.STEPS = []        # decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(soorten)   # number of soorten is number of classes

cfg.SOLVER.CHECKPOINT_PERIOD = round(cfg.SOLVER.MAX_ITER/10) # save checkpoints to watch for underfitting
cfg.INPUT.MASK_FORMAT = "bitmask" # Set the MASK_FORMAT to bitmask

# Necessary to train on images with no objects/birds
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.OUTPUT_DIR = "outputs/model" # checkpoints will be saved here

# Resize images
cfg.INPUT.MIN_SIZE_TRAIN = (445, 640, 890) # 445 and 890 PHZ pixels used for 5m and 10m training, 640 used in SAHI
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.INPUT.MIN_SIZE_TEST = 640 # used in SAHI (smart?)

# Other custom parameters
# cfg.MODEL.RPN.IN_FEATURES, cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, cfg.MODEL.ANCHOR_GENERATOR.SIZES
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION, cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS

# Check if a GPU is available
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cuda' # Otherwise will get Runtime error as no NVIDIA
else:
    cfg.MODEL.DEVICE = 'cpu'

# ==============================================================

# Hooks to log (only doing that for Azure)

# ==============================================================

# Log loss per iteration
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
        val_list = []
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
                
# Hook that logs every metric
class MLflowHook(HookBase):

# from https://philenius.github.io/machine%20learning/2022/01/09/how-to-log-artifacts-metrics-and-parameters-of-your-detectron2-model-training-to-mlflow.html

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])     

# Hook that logs loss per epoch
class LossEvalHook(HookBase):
    
# from https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e

    def __init__(self, cfg, data_loader, epoch_iter, is_validation=False):
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = self.cfg.DATASETS.TEST if is_validation else self.cfg.DATASETS.TRAIN
        self._data_loader = iter(build_detection_train_loader(self.cfg))
        self._data_loader = data_loader
        self._period = epoch_iter # 20
        self.is_validation = is_validation
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss done {}/{}. {:.4f} s / img.".format(
                        idx + 1, total, seconds_per_img
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        
        if self.is_validation:
            mlflow.log_metric('vali_loss', mean_loss)
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
        else:
            mlflow.log_metric('train_loss', mean_loss)
            self.trainer.storage.put_scalar('training_loss', mean_loss)
        
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self.trainer.model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

# ==============================================================

# Train

# ==============================================================         

# Create trainer
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.RandomBrightness(0.5, 2),
                T.RandomContrast(0.5, 2),
                T.RandomSaturation(0.5, 2),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            ]
        )
        return build_detection_train_loader(cfg, mapper=mapper)

trainer = MyTrainer(cfg)

# Add hooks
data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg,True))
epoch_iter = int((train_length * 0.9)/cfg.SOLVER.IMS_PER_BATCH) # number of iterations per epoch
val_hook = LossEvalHook(cfg, data_loader, epoch_iter, is_validation=True)
train_hook = LossEvalHook(cfg, data_loader, epoch_iter, is_validation=False)
ml_flow_hook = MLflowHook()
trainer.register_hooks([val_hook, train_hook, ml_flow_hook])

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

