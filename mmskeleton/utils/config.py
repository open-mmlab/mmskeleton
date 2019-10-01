import os
from mmcv import Config as BaseConfig
from mmskeleton.version import mmskl_home


class Config(BaseConfig):
    @staticmethod
    def fromfile(filename):
        try:
            return BaseConfig.fromfile(filename)
        except:
            return BaseConfig.fromfile(os.path.join(mmskl_home, filename))
