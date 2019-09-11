from mmcv.runner import load_checkpoint as mmcv_load_checkpoint

mmskeleton_model_urls = {
    'st_gcn/kinetics-skeleton': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.kinetics-6fa43f73.pth",
    'st_gcn/ntu-xsub': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xsub-300b57d4.pth",
    'st_gcn/ntu-xview': "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xview-9ba67746.pth",
    'mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
    'pose_estimation/pose_hrnet_w32_256x192_official': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/pose_estimation/pose_hrnet_w32_256x192_official-e6e09ec4.pth',
}  # yapf: disable


def load_checkpoint(model, filename, *args, **kwargs):
    if filename.startswith('mmskeleton://'):
        model_name = filename[13:]
        model_url = (mmskeleton_model_urls[model_name])
        checkpoint = mmcv_load_checkpoint(model, model_url, *args, **kwargs)
        return checkpoint
