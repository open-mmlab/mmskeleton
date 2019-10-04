import os
import json
import mmcv
import numpy as np
import ntpath
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager
from mmskeleton.utils import cache_checkpoint
from mmskeleton.processor.apis import save_batch_image_with_joints
from mmcv.utils import ProgressBar

pose_estimators = dict()


def worker(inputs, results, gpu, detection_cfg, estimation_cfg):
    worker_id = current_process()._identity[0] - 1
    global pose_estimators
    if worker_id not in pose_estimators:
        pose_estimators[worker_id] = init_pose_estimator(detection_cfg,
                                                         estimation_cfg,
                                                         device=gpu)
    while True:
        idx, image = inputs.get()

        # end signal
        if image is None:
            return

        res = inference_pose_estimator(pose_estimators[worker_id], image)
        res['frame_index'] = idx
        results.put(res)


def build(detection_cfg,
          estimation_cfg,
          video_dir,
          out_dir,
          gpus=1,
          worker_per_gpu=1,
          video_max_length=10000,
          category_annotation=None):

    cache_checkpoint(detection_cfg.checkpoint_file)
    cache_checkpoint(estimation_cfg.checkpoint_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if category_annotation is None:
        video_categories = dict()
    else:
        with open(category_annotation) as f:
            video_categories = json.load(f)['annotations']

    inputs = Manager().Queue(video_max_length)
    results = Manager().Queue(video_max_length)

    num_worker = gpus * worker_per_gpu
    procs = []
    for i in range(num_worker):
        p = Process(target=worker,
                    args=(inputs, results, i % gpus, detection_cfg,
                          estimation_cfg))
        procs.append(p)
        p.start()

    video_file_list = os.listdir(video_dir)
    prog_bar = ProgressBar(len(video_file_list))
    for video_file in video_file_list:

        reader = mmcv.VideoReader(os.path.join(video_dir, video_file))
        video_frames = reader[:video_max_length]
        annotations = []
        num_keypoints = -1

        for i, image in enumerate(video_frames):
            inputs.put((i, image))

        for i in range(len(video_frames)):
            t = results.get()
            if not t['has_return']:
                continue

            num_person = len(t['joint_preds'])
            assert len(t['person_bbox']) == num_person

            for j in range(num_person):
                keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                    t['joint_preds'][j].round().astype(int).tolist(),
                    t['joint_scores'][j].tolist())]
                num_keypoints = len(keypoints)
                person_info = dict(person_bbox=t['person_bbox']
                                   [j].round().astype(int).tolist(),
                                   frame_index=t['frame_index'],
                                   id=j,
                                   person_id=None,
                                   keypoints=keypoints)

                annotations.append(person_info)

        # output results
        annotations = sorted(annotations, key=lambda x: x['frame_index'])
        category_id = video_categories[video_file][
            'category_id'] if video_file in video_categories else -1
        info = dict(video_name=video_file,
                    resolution=reader.resolution,
                    num_frame=len(video_frames),
                    num_keypoints=num_keypoints,
                    keypoint_channels=['x', 'y', 'score'],
                    version='1.0')
        video_info = dict(info=info,
                          category_id=category_id,
                          annotations=annotations)
        with open(os.path.join(out_dir, video_file + '.json'), 'w') as f:
            json.dump(video_info, f)

        prog_bar.update()

    # send end signals
    for p in procs:
        inputs.put((-1, None))
    # wait to finish
    for p in procs:
        p.join()

    print('\nBuild skeleton dataset to {}.'.format(out_dir))
    return video_info