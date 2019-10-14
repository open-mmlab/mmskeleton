import os
import numpy as np
import json
import torch


class SkeletonLoader(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to data folder
        num_track: number of skeleton output
        pad_value: the values for padding missed joint
        repeat: times of repeating the dataset
    """
    def __init__(self, data_dir, num_track=1, repeat=1, num_keypoints=-1):

        self.data_dir = data_dir
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
        ] * repeat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        with open(self.files[index]) as f:
            data = json.load(f)

        info = data['info']
        annotations = data['annotations']
        num_frame = info['num_frame']
        num_keypoints = info[
            'num_keypoints'] if self.num_keypoints <= 0 else self.num_keypoints
        channel = info['keypoint_channels']
        num_channel = len(channel)

        # get data
        data['data'] = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()

        return data
