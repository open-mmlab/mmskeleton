import torch
import mmcv
import logging
import torch.multiprocessing as mp
import numpy as np
import cv2
from time import time
from mmcv.utils import ProgressBar
from .pose_demo import inference as pose_inference
from mmskeleton.utils import call_obj, load_checkpoint


def init_recognizer(recognition_cfg, device):
    model = call_obj(**(recognition_cfg.model_cfg))
    load_checkpoint(model,
                    recognition_cfg.checkpoint_file,
                    map_location=device)
    return model


def inference(detection_cfg,
              estimation_cfg,
              recognition_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

    recognizer = init_recognizer(recognition_cfg, 0)
    # import IPython
    # IPython.embed()
    resolution = mmcv.VideoReader(video_file).resolution
    results = pose_inference(detection_cfg, estimation_cfg, video_file, gpus,
                             worker_per_gpu)

    seq = np.zeros((1, 3, len(results), 17, 1))
    for i, r in enumerate(results):
        if r['joint_preds'] is not None:
            seq[0, 0, i, :, 0] = r['joint_preds'][0, :, 0] / resolution[0]
            seq[0, 1, i, :, 0] = r['joint_preds'][0, :, 1] / resolution[1]
            seq[0, 2, i, :, 0] = r['joint_scores'][0, :, 0]

    import IPython
    IPython.embed()

    return results