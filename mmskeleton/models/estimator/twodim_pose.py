import torch.nn as nn
from .base import BaseEstimator
from mmskeleton.utils.importer import call_obj
class TwoDimPoseEstimator(BaseEstimator):
    def __init__(self,
                 backbone,
                 neck=None,
                 skeleton_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoDimPoseEstimator,self).__init__()
        self.backbone = call_obj(**backbone)
        if neck is not None:
            self.neck = call_obj(**neck)
        self.skeleton_head = call_obj(**skeleton_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(TwoDimPoseEstimator, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.skeleton_head.init_weights()


    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.skeleton_head(x[0])
        return outs

    def forward_train(self,
                      image,
                      meta,
                      targets,
                      target_weights
                      ):

        x =self.extract_feat(image)
        outs = self.skeleton_head(x)
        loss_inputs = [outs, targets, target_weights]
        losses = self.skeleton_head.loss(
            *loss_inputs)

        return losses

    def forward_test(self, image, **kwargs):
        x =self.extract_feat(image)
        outs = self.skeleton_head(x)
        return outs





