## Start Action Recognition

This repository holds the codebase for the paper:

**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

<div align="center">
    <img src="../demo/recognition/pipeline.png">
</div>


### Data Preparation

We experimented on two skeleton-based action recognition datasts: **Kinetics-skeleton** and **NTU RGB+D**.
Before training and testing, for the convenience of fast data loading,
the datasets should be converted to proper format.
Download the pre-processed data from
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
and extract files with
```
cd st-gcn
unzip <path to st-gcn-processed-data.zip>
```

## Training

To train a ST-GCN model, run

``` shell
python run.py configs/st_gcn/recognition/st_gcn/$DATASET/train.yaml [optional arguments]
```
where the `$DATASET` must be `ntu-xsub`, `ntu-xview` or `kinetics-skeleton`.

All outputs (log files and ) will be saved to the working directory, which is specified by `work_dir` in the config file.
If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`. Check all optional arguments via adding `-h` argument.

Model evaluation can be achieved by:
``` shell
python run.py configs/st_gcn/recognition/st_gcn/$DATASET/train.yaml --checkpoint $CHECKPOINT_FILE [optional arguments]
```

### Results
The expected **Top-1** **accuracy** of provided models are shown here:

| Model| Kinetics-<br>skeleton (%)|NTU RGB+D <br> Cross View (%) |NTU RGB+D <br> Cross Subject (%) |
| :------| :------: | :------: | :------: |
|**ST-GCN** (Ours)| **31.6**| **88.8** | **81.6** |
<!-- |Baseline[1]| 20.3    | 83.1     |  74.3    | -->

<!-- [1] Kim, T. S., and Reiter, A. 2017. Interpretable 3d human action analysis with temporal convolutional networks. In BNMW CVPRW. -->