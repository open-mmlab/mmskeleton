import torch
import mmcv
import logging
import torch.multiprocessing as mp
import numpy as np
import cv2
from time import time
from mmskeleton.datasets.utils.video_demo import VideoDemo
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmskeleton.processor.apis import init_twodimestimator, inference_twodimestimator, save_batch_image_with_joints
from mmcv.utils import ProgressBar
import os
logger = logging.getLogger()
import sys
sys.setrecursionlimit(1000000)

def save(image,
         det_image,
         pred,
         name
         ):
    batch_size = pred.shape[0]
    num_joints = pred.shape[1]
    cimage = np.expand_dims(image, axis=0)
    cimage = torch.from_numpy(cimage)
    pred = torch.from_numpy(pred)
    cimage= cimage.permute(0, 3, 1, 2)
    pred_vis = torch.ones((batch_size, num_joints, 1))
    ndrr = save_batch_image_with_joints( cimage, pred, pred_vis)
    mask = ndrr[:,:, 0]  == 255
    print(mask.sum())
    mask = np.expand_dims(mask, axis=2)
    out = ndrr * mask + det_image * (1 - mask)
    mmcv.imwrite(out, name)


def worker(
  video_file,
  index,
  detection_cfg,
  skeleton_cfg,
  skeleon_data_cfg,
  device,
  result_queue
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    video_frames = mmcv.VideoReader(video_file)
    # build model
    logger.info('Begin to build detection model')
    beign_time = time()
    detection_model_file = detection_cfg.model_cfg
    detection_checkpoint_file = detection_cfg.checkpoint_file
    detection_model = init_detector(detection_model_file,
                                    detection_checkpoint_file,
                                    device='cpu')

    end_time = time()
    logger.info('Detection model has been built successfully, costing: {}'.format(end_time - beign_time))
    # build skeleton model
    logger.info('Begin to build estimation model')
    beign_time = time()
    skeleton_model_file = skeleton_cfg.model_cfg
    skeletion_checkpoint_file = skeleton_cfg.checkpint_file
    skeleton_model = init_twodimestimator(skeleton_model_file,
                                          skeletion_checkpoint_file,
                                          device='cpu')
    end_time = time()
    logger.info('Estimation model has been built successfully, costing: {}'.format(end_time - beign_time))
    detection_model = detection_model.cuda()

    skeleton_model = skeleton_model.cuda()
    for idx in index:
        skeleton_result = dict()
        image = video_frames[idx]
        draw_image = image.copy()
        bbox_result = inference_detector(detection_model, image)

        person_bbox, labels = VideoDemo.bbox_filter(bbox_result, detection_cfg.bbox_thre)

        if len(person_bbox) > 0:
            person, meta = VideoDemo.skeleton_preprocess(image[:,:,::-1], person_bbox, skeleon_data_cfg)
            preds, maxvals = inference_twodimestimator(skeleton_model, person.cuda(), meta, True)
            results = VideoDemo.skeleton_postprocess(preds, maxvals, meta)
            if skeleon_data_cfg.save_image:
                file = skeleon_data_cfg.save_dir + '{}.png'.format(idx)
                mmcv.imshow_det_bboxes(
                    draw_image,
                    person_bbox,
                    labels,
                    detection_model.CLASSES,
                    score_thr=detection_cfg.bbox_thre,
                    show=False,
                    wait_time=0)
                save(image, draw_image, results, file)

        else:
            preds, maxvals = None, None
            if skeleon_data_cfg.save_image:
                file = skeleon_data_cfg.save_dir + '{}.png'.format(idx)
                mmcv.imwrite(image, file)
        skeleton_result['frame_index'] = idx
        skeleton_result['position_preds'] = preds
        skeleton_result['position_maxvals'] = maxvals
        result_queue.put(skeleton_result)



def inference(
        detection_cfg,
        skeleton_cfg,
        dataset_cfg,
        gpus=1,
):
    # get frame num
    video_file = dataset_cfg.video_file
    video_name = video_file.strip('/n').split('/')[-1]
    video_frames = mmcv.VideoReader(video_file)
    num_frames = len(video_frames)
    del video_frames

    data_cfg = skeleton_cfg.data_cfg
    if data_cfg.save_image:
        if not os.path.exists(data_cfg.save_dir):
            os.mkdir(data_cfg.save_dir)

    # multiprocess settings
    context = mp.get_context('spawn')
    result_queue = context.Queue(num_frames)
    stride = int(np.ceil(num_frames/ gpus))
    procs = []
    for d in range(gpus):
        e_record = min((d + 1) * stride, num_frames)
        shred_list = list(range(d * stride, e_record))
        p = context.Process(target=worker,
                            args=(video_file,
                                  shred_list,
                                  detection_cfg,
                                  skeleton_cfg,
                                  data_cfg,
                                  d,
                                  result_queue))
        p.start()
        procs.append(p)
    all_result = []
    prog_bar = ProgressBar(num_frames)
    for _ in range(num_frames):
        t = result_queue.get()
        all_result.append(t)
        prog_bar.update()
    for p in procs:
        p.join()
    if len(all_result) == num_frames:
        mmcv.frames2video(data_cfg.save_dir, video_name, filename_tmpl='{:01d}.png')