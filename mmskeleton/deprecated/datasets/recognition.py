import os
import numpy as np
import json
import torch
from .utils import skeleton


class SkeletonDataset(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to data folder
        random_choose: If true, randomly choose a portion of the input sequence
        random_move: If true, randomly perfrom affine transformation
        window_size: The length of the output sequence
        repeat: times of repeating the dataset
        data_subscripts: subscript expression of einsum operation.
            In the default case, the shape of output data is `(channel, vertex, frames, person)`.
            To permute the shape to `(channel, frames, vertex, person)`,
            set `data_subscripts` to 'cvfm->cfvm'.
    """
    def __init__(self,
                 data_dir,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 num_track=1,
                 data_subscripts=None,
                 repeat=1):

        self.data_dir = data_dir
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.num_track = num_track
        self.data_subscripts = data_subscripts
        self.files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
        ] * repeat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        with open(self.files[index]) as f:
            data = json.load(f)

        resolution = data['info']['resolution']
        category_id = data['category_id']
        annotations = data['annotations']
        num_frame = data['info']['num_frame']
        num_keypoints = data['info']['num_keypoints']
        channel = data['info']['keypoint_channels']
        num_channel = len(channel)

        # get data
        data = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data[:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()

        # normalization
        if self.normalization:
            for i, c in enumerate(channel):
                if c == 'x':
                    data[i] = data[i] / resolution[0] - 0.5
                if c == 'y':
                    data[i] = data[i] / resolution[1] - 0.5
                if c == 'score' or c == 'visibility':
                    mask = (data[i] == 0)
                    for j in range(num_channel):
                        if c != j:
                            data[j][mask] = 0

        # permute
        if self.data_subscripts is not None:
            data = np.einsum(self.data_subscripts, data)

        # augmentation
        if self.random_choose:
            data = skeleton.random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = skeleton.auto_pading(data, self.window_size)
        if self.random_move:
            data = skeleton.random_move(data)

        return data, category_id