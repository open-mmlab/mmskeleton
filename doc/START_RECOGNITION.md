## Start Action Recognition Using ST-GCN

This repository holds the codebase for the paper:

**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

<div align="center">
    <img src="../demo/recognition/pipeline.png">
</div>


### Data Preparation

We experimented on two skeleton-based action recognition datasts: **Kinetics-skeleton** and **NTU RGB+D**.
Before training and testing, for the convenience of fast data loading,
the datasets should be converted to the proper format.
Please download the pre-processed data from
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
and extract files with
```
cd st-gcn
unzip <path to st-gcn-processed-data.zip>
```

If you want to process data by yourself, please refer to [SKELETON_DATA.md](./SKELETON_DATA.md) for more details.

### Evaluate Pretrained Models

The evaluation of pre-trained models on three datasets can be achieved by:

``` shell
mmskl configs/recognition/st_gcn_aaai18/$DATASET/test.yaml
```
where the `$DATASET` must be `ntu-rgbd-xsub`, `ntu-rgbd-xview` or `kinetics-skeleton`.
Models will be downloaded automatically before testing.
The expected accuracies are shown here:

| Dataset                 | Top-1 Accuracy (%) | Top-5 Accuracy (%) |                                                      Download                                                      |
|:------------------------|:------------------:|:------------------:|:------------------------------------------------------------------------------------------------------------------:|
| Kinetics-skeleton       |       31.60        |       53.68        | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.kinetics-6fa43f73.pth)  |
| NTU RGB+D Cross View    |       88.76        |       98.83        | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xview-9ba67746.pth) |
| NTU RGB+D Cross Subject |       81.57        |       96.85        | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xsub-300b57d4.pth)  |


### Training

To train a ST-GCN model, run

``` shell
mmskl configs/recognition/st_gcn_aaai18/$DATASET/train.yaml [optional arguments]
```

The usage of optional arguments can be checked via adding `--help` argument.
All outputs (log files and ) will be saved to the default working directory.
That can be changed by modifying the configuration file
or adding a optional argument `--work_dir $WORKING_DIRECTORY` in the command line.

After that, evaluate your models by:

``` shell
mmskl configs/recognition/st_gcn_aaai18/$DATASET/test.yaml --checkpoint $CHECKPOINT_FILE
```
