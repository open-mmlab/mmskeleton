import torch
import mmcv
import numpy as np
import cv2
import os
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager
from mmskeleton.utils import cache_checkpoint, third_party
from mmcv.utils import ProgressBar


def render(image, pred, person_bbox, bbox_thre=0):
    if pred is None:
        return image

    mmcv.imshow_det_bboxes(image,
                           person_bbox,
                           np.zeros(len(person_bbox)).astype(int),
                           class_names=['person'],
                           score_thr=bbox_thre,
                           show=False,
                           wait_time=0)

    for person_pred in pred:
        for joint_pred in person_pred:
            cv2.circle(image, (int(joint_pred[0]), int(joint_pred[1])), 2,
                       [255, 0, 0], 2)

    return np.uint8(image)


pose_estimators = dict()


def worker(inputs, results, gpu, detection_cfg, estimation_cfg, render_image):
    worker_id = current_process()._identity[0] - 1
    global pose_estimators
    if worker_id not in pose_estimators:
        pose_estimators[worker_id] = init_pose_estimator(detection_cfg,
                                                         estimation_cfg,
                                                         device=gpu)
    while not inputs.empty():
        try:
            idx, image = inputs.get_nowait()
        except:
            return

        res = inference_pose_estimator(pose_estimators[worker_id], image)
        res['frame_index'] = idx

        if render_image:
            res['render_image'] = render(image, res['joint_preds'],
                                         res['person_bbox'],
                                         detection_cfg.bbox_thre)

        results.put(res)


def inference(detection_cfg,
              estimation_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

    if not third_party.is_exist('mmdet'):
        print(
            '\nERROR: This demo requires mmdet for detecting bounding boxes of person. Please install mmdet first.'
        )
        exit(1)

    video_frames = mmcv.VideoReader(video_file)
    all_result = []
    print('\nPose estimation:')

    # case for single process
    if gpus == 1 and worker_per_gpu == 1:
        model = init_pose_estimator(detection_cfg, estimation_cfg, device=0)
        prog_bar = ProgressBar(len(video_frames))
        for i, image in enumerate(video_frames):
            res = inference_pose_estimator(model, image)
            res['frame_index'] = i
            if save_dir is not None:
                res['render_image'] = render(image, res['joint_preds'],
                                             res['person_bbox'],
                                             detection_cfg.bbox_thre)
            all_result.append(res)
            prog_bar.update()

    # case for multi-process
    else:
        cache_checkpoint(detection_cfg.checkpoint_file)
        cache_checkpoint(estimation_cfg.checkpoint_file)
        num_worker = gpus * worker_per_gpu
        procs = []
        inputs = Manager().Queue(len(video_frames))
        results = Manager().Queue(len(video_frames))

        for i, image in enumerate(video_frames):
            inputs.put((i, image))

        for i in range(num_worker):
            p = Process(target=worker,
                        args=(inputs, results, i % gpus, detection_cfg,
                              estimation_cfg, save_dir is not None))
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

    # sort results
    all_result = sorted(all_result, key=lambda x: x['frame_index'])

    # generate video
    if (len(all_result) == len(video_frames)) and (save_dir is not None):
        print('\n\nGenerate video:')
        video_name = video_file.strip('/n').split('/')[-1]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        video_path = os.path.join(save_dir, video_name)
        vwriter = cv2.VideoWriter(video_path,
                                  mmcv.video.io.VideoWriter_fourcc(*('mp4v')),
                                  video_frames.fps, video_frames.resolution)
        prog_bar = ProgressBar(len(video_frames))
        for r in all_result:
            vwriter.write(r['render_image'])
            prog_bar.update()
        vwriter.release()
        print('\nVideo was saved to {}'.format(video_path))

    return all_result