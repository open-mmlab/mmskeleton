from .twodim_pose import TwoDimPoseEstimator

class HRPoseEstimator(TwoDimPoseEstimator):
    def __init__(self,
                 backbone,
                 neck = None,
                 skeleton_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None):
        super(HRPoseEstimator, self).\
            __init__(backbone, neck, skeleton_head, train_cfg,
                     test_cfg, pretrained)

