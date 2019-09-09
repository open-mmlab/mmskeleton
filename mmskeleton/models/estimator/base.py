import logging
from abc import ABCMeta, abstractmethod
import torch.nn as nn

class BaseEstimator(nn.Module):
    """Base class for pose estimation"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseEstimator, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, input, meta, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, input, meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, input, meta, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, input, **kwargs):
        pass

    def forward(self, image, meta = None, targets = None, target_weights = None, return_loss=True, **kwargs):
        if return_loss:
            return  self.forward_train(image, meta, targets, target_weights, **kwargs)
        else:
            return  self.forward_test(image, **kwargs)
