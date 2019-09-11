import torch.nn as nn
from mmskeleton.utils.importer import call_obj


class SimpleSkeletonHead(nn.Module):
    def __init__(self,
                 num_convs,
                 in_channels,
                 embed_channels=None,
                 kernel_size=None,
                 num_joints=None,
                 reg_loss=dict(name='JointsMSELoss', use_target_weight=False)):
        super(SimpleSkeletonHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.skeleton_reg = self.make_layers()
        self.reg_loss = call_obj(**reg_loss)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def make_layers(self):

        assert isinstance(self.embed_channels, list) or isinstance(self.embed_channels, int) or \
               self.embed_channels is None
        assert isinstance(self.kernel_size, list) or isinstance(
            self.kernel_size, int)

        if isinstance(self.embed_channels, list):
            assert len(self.embed_channels) == self.num_convs - 1

        if isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == self.num_convs

        module_list = []

        for i in range(self.num_convs):
            if i == 0:
                in_channels = self.in_channels

            if i < self.num_convs - 1:
                if isinstance(self.embed_channels, list):
                    out_channels = self.embed_channels[i]
                elif isinstance(self.embed_channels, int):
                    out_channels = self.embed_channels
            elif (i == self.num_convs - 1) or isinstance(
                    self.embed_channels, None):
                out_channels = self.num_joints

            if isinstance(self.kernel_size, list):
                kernel_size = self.kernel_size[i]
            else:
                kernel_size = self.kernel_size

            padding = kernel_size // 2
            module_list.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=1))

            in_channels = out_channels

        return nn.Sequential(*module_list)

    def forward(self, x):
        reg_pred = self.skeleton_reg(x[0])
        return reg_pred

    def loss(self, outs, targets, target_weights):
        losses = dict()
        losses['reg_loss'] = self.reg_loss(outs, targets, target_weights)

        return losses
