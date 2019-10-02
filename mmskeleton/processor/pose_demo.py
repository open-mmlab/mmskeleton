import torch
import mmcv
import logging
import torch.multiprocessing as mp
import numpy as np
import cv2
from time import time
from mmskeleton.utils import cache_checkpoint, get_mmskeleton_url
from mmskeleton.datasets.utils.video_demo import VideoDemo
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmskeleton.processor.apis import init_twodimestimator, inference_twodimestimator, save_batch_image_with_joints
from mmcv.utils import ProgressBar
import os
logger = logging.getLogger()
import sys
sys.setrecursionlimit(1000000)
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import Pool, current_process, Process, Manager

pose_estimators = dict()


def worker(gpu,
           detection_cfg,
           skeleton_cfg,
           video_frames,
           results,
           frame_index=-1):
    worker_id = current_process()._identity[0] - 1
    if worker_id not in pose_estimators:
        pose_estimators[worker_id] = init_pose_estimator(detection_cfg,
                                                         skeleton_cfg,
                                                         device=gpu)

    for image in video_frames:
        res = inference_pose_estimator(pose_estimators[worker_id], image)
        res['frame_index'] = frame_index
        results.put(res)


def inference(detection_cfg,
              skeleton_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

    video_frames = mmcv.VideoReader(video_file)
    all_result = []

    # case for single process
    if gpus == 1 and worker_per_gpu == 1:
        model = init_pose_estimator(detection_cfg, skeleton_cfg, device=0)
        prog_bar = ProgressBar(len(video_frames))
        for i, image in enumerate(video_frames):
            res = inference_pose_estimator(model, image)
            res['frame_index'] = i
            all_result.append(res)
            prog_bar.update()

    # case for multi-process
    else:
        num_worker = gpus * worker_per_gpu
        procs = []
        results = Manager().Queue(len(video_frames))
        for i in range(num_worker):
            frames = [
                f for j, f in enumerate(video_frames) if j % num_worker == i
            ]
            p = Process(target=worker,
                        args=(i % gpus, detection_cfg, skeleton_cfg, frames,
                              results))
            procs.append(p)
            p.start()
        for i in range(len(video_frames)):
            t = results.get()
            all_result.append(t)
            if 'prog_bar' not in locals():
                prog_bar = ProgressBar(len(video_frames))
            prog_bar.update()
        for p in procs:
            p.join()

    # generate video
    if len(all_result) == len(video_frames) and save_dir is not None:
        print('\n\nGenerate video:')
        video_name = video_file.strip('/n').split('/')[-1]
        video_path = os.path.join(save_dir, video_name)
        img_dir = os.path.join(save_dir, '{}.img'.format(video_name))
        mmcv.frames2video(img_dir, video_path, filename_tmpl='{:01d}.png')
        print('Video was saved to {}'.format(video_path))

    return all_result