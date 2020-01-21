import os
import cv2
import torch
import torchvision
import math
import numpy as np

from mmskeleton.utils import call_obj
from mmskeleton.utils import load_checkpoint
from .utils.infernce_utils import get_final_preds
from mmcv.utils import Config
from collections import OrderedDict
from mmskeleton.datasets.utils.coco_transform import flip_back
from mmskeleton.datasets.utils.video_demo import VideoDemo
from mmskeleton.utils import get_mmskeleton_url

flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
              [15, 16]]


def init_twodimestimator(config, checkpoint=None, device='cpu'):
    if isinstance(config, str):
        config = Config.fromfile(config)
        config = config.processor_cfg
    elif isinstance(config, OrderedDict):
        config = config
    else:
        raise ValueError(
            'Input config type is: {}, expect "str" or "Orderdict"'.format(
                type(config)))
    model_cfg = config.model_cfg

    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location=device)
    model.to(device)
    model = model.eval()

    return model


def inference_twodimestimator(model, input, meta, flip=False):
    with torch.no_grad():
        outputs = model.forward(input, return_loss=False)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
        if flip:
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            outputs_flipped = model(input_flipped, return_loss=False)
            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped
            output_flipped = flip_back(output_flipped.detach().cpu().numpy(),
                                       flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
            # feature is not aligned, shift flipped heatmap for higher accuracy

            output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        preds, maxvals = get_final_preds(True,
                                         output.detach().cpu().numpy(), c, s)

    return preds, maxvals


def save_batch_image_with_joints(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 nrow=8,
                                 padding=2):

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy() * 0

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                               [255, 0, 0], 2)
            k = k + 1
    return ndarr