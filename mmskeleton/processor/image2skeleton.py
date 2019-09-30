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


def save(image, det_image, pred, name):
    batch_size = pred.shape[0]
    num_joints = pred.shape[1]
    cimage = np.expand_dims(image, axis=0)
    cimage = torch.from_numpy(cimage)
    pred = torch.from_numpy(pred)
    cimage = cimage.permute(0, 3, 1, 2)
    pred_vis = torch.ones((batch_size, num_joints, 1))
    ndrr = save_batch_image_with_joints(cimage, pred, pred_vis)
    mask = ndrr[:, :, 0] == 255
    mask = np.expand_dims(mask, axis=2)
    out = ndrr * mask + det_image * (1 - mask)
    mmcv.imwrite(out, name)


def worker(video_file, index, detection_cfg, skeleton_cfg, skeleon_data_cfg,
           device, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    video_frames = mmcv.VideoReader(video_file)

    # load model
    detection_model_file = detection_cfg.model_cfg
    detection_checkpoint_file = get_mmskeleton_url(
        detection_cfg.checkpoint_file)
    detection_model = init_detector(detection_model_file,
                                    detection_checkpoint_file,
                                    device='cpu')
    skeleton_model_file = skeleton_cfg.model_cfg
    skeletion_checkpoint_file = skeleton_cfg.checkpoint_file
    skeleton_model = init_twodimestimator(skeleton_model_file,
                                          skeletion_checkpoint_file,
                                          device='cpu')

    detection_model = detection_model.cuda()
    skeleton_model = skeleton_model.cuda()

    for idx in index:
        skeleton_result = dict()
        image = video_frames[idx]
        draw_image = image.copy()
        bbox_result = inference_detector(detection_model, image)

        person_bbox, labels = VideoDemo.bbox_filter(bbox_result,
                                                    detection_cfg.bbox_thre)

        if len(person_bbox) > 0:
            person, meta = VideoDemo.skeleton_preprocess(
                image[:, :, ::-1], person_bbox, skeleon_data_cfg)
            preds, maxvals = inference_twodimestimator(skeleton_model,
                                                       person.cuda(), meta,
                                                       True)
            results = VideoDemo.skeleton_postprocess(preds, maxvals, meta)
            if skeleon_data_cfg.save_video:
                file = os.path.join(skeleon_data_cfg.img_dir,
                                    '{}.png'.format(idx))
                mmcv.imshow_det_bboxes(draw_image,
                                       person_bbox,
                                       labels,
                                       detection_model.CLASSES,
                                       score_thr=detection_cfg.bbox_thre,
                                       show=False,
                                       wait_time=0)
                save(image, draw_image, results, file)

        else:
            preds, maxvals = None, None
            if skeleon_data_cfg.save_video:
                file = os.path.join(skeleon_data_cfg.img_dir,
                                    '{}.png'.format(idx))
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
        worker_per_gpu=1,
):
    # get frame num
    video_file = dataset_cfg.video_file
    video_name = video_file.strip('/n').split('/')[-1]
    video_frames = mmcv.VideoReader(video_file)
    num_frames = len(video_frames)
    del video_frames

    data_cfg = skeleton_cfg.data_cfg
    if data_cfg.save_video:
        data_cfg.img_dir = os.path.join(data_cfg.save_dir,
                                        '{}.img'.format(video_name))

        if os.path.exists(data_cfg.img_dir):
            import shutil
            shutil.rmtree(data_cfg.img_dir)

        os.makedirs(data_cfg.img_dir)

    # cache model checkpoints
    cache_checkpoint(detection_cfg.checkpoint_file)
    cache_checkpoint(skeleton_cfg.checkpoint_file)

    # multiprocess settings
    context = mp.get_context('spawn')
    result_queue = context.Queue(num_frames)
    procs = []
    for w in range(gpus * worker_per_gpu):
        shred_list = list(range(w, num_frames, gpus * worker_per_gpu))
        p = context.Process(target=worker,
                            args=(video_file, shred_list, detection_cfg,
                                  skeleton_cfg, data_cfg, w % gpus,
                                  result_queue))
        p.start()
        procs.append(p)
    all_result = []
    print('\nPose estimation start:')
    prog_bar = ProgressBar(num_frames)
    for i in range(num_frames):
        t = result_queue.get()
        all_result.append(t)
        prog_bar.update()
    for p in procs:
        p.join()
    if len(all_result) == num_frames and data_cfg.save_video:
        print('\n\nGenerate video:')
        video_path = os.path.join(data_cfg.save_dir, video_name)
        mmcv.frames2video(data_cfg.img_dir,
                          video_path,
                          filename_tmpl='{:01d}.png')
        print('Video was saved to {}'.format(video_path))

    import IPython
    IPython.embed()
