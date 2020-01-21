import os
import numpy as np
import torch
import mmcv

from mmskeleton.datasets.utils.video_demo import VideoDemo
from mmskeleton.utils import get_mmskeleton_url
from mmskeleton.processor.apis import init_twodimestimator, inference_twodimestimator

import mmdet.apis


def init_pose_estimator(detection_cfg, estimation_cfg, device=None):

    detection_model_file = detection_cfg.model_cfg
    detection_checkpoint_file = get_mmskeleton_url(
        detection_cfg.checkpoint_file)
    detection_model = mmdet.apis.init_detector(detection_model_file,
                                               detection_checkpoint_file,
                                               device='cpu')

    skeleton_model_file = estimation_cfg.model_cfg
    skeletion_checkpoint_file = estimation_cfg.checkpoint_file
    skeleton_model = init_twodimestimator(skeleton_model_file,
                                          skeletion_checkpoint_file,
                                          device='cpu')

    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        detection_model = detection_model.cuda()
        skeleton_model = skeleton_model.cuda()

    pose_estimator = (detection_model, skeleton_model, detection_cfg,
                      estimation_cfg)
    return pose_estimator


def inference_pose_estimator(pose_estimator, image):
    detection_model, skeleton_model, detection_cfg, estimation_cfg = pose_estimator
    bbox_result = mmdet.apis.inference_detector(detection_model, image)
    person_bbox, labels = VideoDemo.bbox_filter(bbox_result,
                                                detection_cfg.bbox_thre)
    if len(person_bbox) > 0:
        has_return = True
        person, meta = VideoDemo.skeleton_preprocess(image[:, :, ::-1],
                                                     person_bbox,
                                                     estimation_cfg.data_cfg)
        preds, maxvals = inference_twodimestimator(skeleton_model,
                                                   person.cuda(), meta, True)

    else:
        has_return = False
        preds, maxvals, meta = None, None, None

    result = dict(joint_preds=preds,
                  joint_scores=maxvals,
                  meta=meta,
                  has_return=has_return,
                  person_bbox=person_bbox)

    return result
