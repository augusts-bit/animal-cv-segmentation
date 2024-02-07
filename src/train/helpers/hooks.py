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
                           
class MLflowHook(HookBase):

    # from https://philenius.github.io/machine%20learning/2022/01/09/how-to-log-artifacts-metrics-and-parameters-of-your-detectron2-model-training-to-mlflow.html
    
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.step_count = 0
        self.total_loss = 0.0

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1]) 
            loss = latest_metrics.get('total_loss', [0.0, 0])[0]
   
            # Accumulate the loss
            self.total_loss += loss
            self.step_count += 1

            # Check if it's time to log the average loss
            if self.step_count % self.log_interval == 0:
                # Calculate the average loss
                avg_loss = self.total_loss / self.log_interval

                # Log the average loss
                mlflow.log_metric(key="epoch_loss", value=avg_loss, step=self.step_count)

                # Reset the counters
                self.total_loss = 0.0

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

        
# val_hook = LossEvalHook(cfg, data_loader, epoch_iter, is_validation=True)
# train_hook = LossEvalHook(cfg, data_loader, epoch_iter, is_validation=False)
# ml_flow_hook = MLflowHook()