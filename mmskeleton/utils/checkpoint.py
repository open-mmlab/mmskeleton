from mmcv.runner import load_checkpoint as mmcv_load_checkpoint
from mmcv.runner.checkpoint import load_url_dist
import urllib


mmskeleton_model_urls = {
    'st_gcn/kinetics-skeleton': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.kinetics-6fa43f73.pth",
    'st_gcn/ntu-xsub': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xsub-300b57d4.pth",
    'st_gcn/ntu-xview': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xview-9ba67746.pth",
    'mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
    'pose_estimation/pose_hrnet_w32_256x192': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/pose_estimation/pose_hrnet_w32_256x192-76ea353b.pth',
    'mmdet/cascade_rcnn_r50_fpn_20e': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth',
}  # yapf: disable


def load_checkpoint(model, filename, *args, **kwargs):
    try:
        filename = get_mmskeleton_url(filename)
        return mmcv_load_checkpoint(model, filename, *args, **kwargs)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise Exception(url_error_message.format(filename)) from e


def get_mmskeleton_url(filename):
    if filename.startswith('mmskeleton://'):
        model_name = filename[13:]
        model_url = (mmskeleton_model_urls[model_name])
        return model_url
    return filename


def cache_checkpoint(filename):
    try:
        filename = get_mmskeleton_url(filename)
        load_url_dist(get_mmskeleton_url(filename))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise Exception(url_error_message.format(filename)) from e


url_error_message = """

==================================================
MMSkeleton fail to load checkpoint from url: 
    {}
Please check your network connection. Or manually download checkpoints according to the instructor:
    https://github.com/open-mmlab/mmskeleton/blob/master/doc/MODEL_ZOO.md
"""