import torch
import mmcv
import numpy as np
import cv2
import os
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager
from mmskeleton.utils import cache_checkpoint
from mmskeleton.processor.apis import save_batch_image_with_joints
from mmcv.utils import ProgressBar


def render(image, pred, person_bbox, bbox_thre):
    if pred is None:
        return image

    det_image = image.copy()
    mmcv.imshow_det_bboxes(det_image,
                           person_bbox,
                           np.zeros(len(person_bbox)).astype(int),
                           class_names=['person'],
                           score_thr=bbox_thre,
                           show=False,
                           wait_time=0)

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
    return np.uint8(out)


pose_estimators = dict()


def worker(inputs, results, gpu, detection_cfg, estimation_cfg, render_image):
    worker_id = current_process()._identity[0] - 1
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
            res['render_image'] = render(image, res['position_preds'],
                                         res['person_bbox'],
                                         detection_cfg.bbox_thre)
        results.put(res)


def inference(detection_cfg,
              estimation_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

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
                res['render_image'] = render(image, res['position_preds'],
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