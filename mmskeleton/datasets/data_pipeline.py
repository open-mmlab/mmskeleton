import os
import numpy as np
import json
import torch
from mmskeleton.utils import call_obj
from .utils import skeleton


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self, data_source, pipeline=[]):

        self.data_source = call_obj(**data_source)
        self.pipeline = pipeline

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        data = index
        data = self.data_source[index]
        for stage_args in self.pipeline:
            data = call_obj(data=data, **stage_args)
        return data
