import torch
import mmcv
import logging
import torch.multiprocessing as mp
import numpy as np
import cv2
from time import time
from mmcv.utils import ProgressBar
from .pose_demo import inference as pose_inference


def inference(detection_cfg,
              estimation_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

    pose = pose_inference(detection_cfg, estimation_cfg, video_file, gpus,
                          worker_per_gpu)

    return pose