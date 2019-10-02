import cv2
import torch
import mmcv
import numpy as np
from .coco_transform import xywh2cs, get_affine_transform
from mmskeleton.ops.nms.nms import oks_nms


class VideoDemo(object):
    def __init__(self, ):
        super(VideoDemo, self).__init__()

    @staticmethod
    def bbox_filter(bbox_result, bbox_thre=0.0):
        # clone from mmdetection

        if isinstance(bbox_result, tuple):
            bbox_result, segm_result = bbox_result
        else:
            bbox_result, segm_result = bbox_result, None

        bboxes = np.vstack(bbox_result)
        bbox_labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(bbox_labels)
        # get bboxes for each person
        person_id = 0
        person_bboxes = bboxes[labels == person_id]
        person_mask = person_bboxes[:, 4] >= bbox_thre
        person_bboxes = person_bboxes[person_mask]
        return person_bboxes, labels[labels == person_id][person_mask]

    @staticmethod
    def skeleton_preprocess(image, bboxes, skeleton_cfg):

        # output collector
        result_list = []
        meta = dict()
        meta['scale'] = []
        meta['rotation'] = []
        meta['center'] = []
        meta['score'] = []

        # preprocess config
        image_size = skeleton_cfg.image_size
        image_width = image_size[0]
        image_height = image_size[1]
        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = skeleton_cfg.pixel_std
        image_mean = skeleton_cfg.image_mean
        image_std = skeleton_cfg.image_std

        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            w, h = x2 - x1, y2 - y1
            center, scale = xywh2cs(x1, y1, h, w, aspect_ratio, pixel_std)
            trans = get_affine_transform(center, scale, 0, image_size)
            transformed_image = cv2.warpAffine(
                image,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)
            # transfer into Torch.Tensor
            transformed_image = transformed_image / 255.0
            transformed_image = transformed_image - image_mean
            transformed_image = transformed_image / image_std
            transformed_image = transformed_image.transpose(2, 0, 1)
            result_list.append(transformed_image)
            # from IPython import embed; embed()
            meta['scale'].append(scale)
            meta['rotation'].append(0)
            meta['center'].append(center)
            meta['score'].append(bbox[4])

        result = torch.from_numpy(np.array(result_list)).float()
        for name, data in meta.items():
            meta[name] = torch.from_numpy(np.array(data)).float()
        return result, meta

    @staticmethod
    def skeleton_postprocess(
            preds,
            max_vals,
            meta,
    ):

        all_preds = np.concatenate((preds, max_vals), axis=-1)
        _kpts = []
        for idx, kpt in enumerate(all_preds):
            center = meta['center'][idx].numpy()
            scale = meta['scale'][idx].numpy()
            area = np.prod(scale * 200, 0)
            score = meta['score'][idx].numpy()
            _kpts.append({
                'keypoints': kpt,
                'center': center,
                'scale': scale,
                'area': area,
                'score': score,
            })
        num_joints = 17
        in_vis_thre = 0.2
        oks_thre = 0.9
        oks_nmsed_kpts = []
        for n_p in _kpts:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, num_joints):
                t_s = n_p['keypoints'][n_jt][2]
                if t_s > in_vis_thre:
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = kpt_score * box_score

            keep = oks_nms([_kpts[i] for i in range(len(_kpts))], oks_thre)

        if len(keep) == 0:
            oks_nmsed_kpts.append(_kpts['keypoints'])
        else:
            oks_nmsed_kpts.append(
                [_kpts[_keep]['keypoints'] for _keep in keep])

        return np.array(oks_nmsed_kpts[0])
