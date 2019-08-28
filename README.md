# MMSkeleton

## Introduction

MMSkeleton is an open source toolbox for skeleton-based human understanding, 
including but not limited to pose estimation, action recognition and skeleton sequence generation.
It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

<p align="center">
    <img src="demo/recognition/demo_video.gif", width="1200">
</p>

## Installation

``` shell
pip install -e .
```

## Run

``` shell
# run a pseudo processor for training.
python tools/run.py configs/pseudo/train.yaml

# train st-gcn.
python tools/run.py configs/st_gcn/train.yaml
# test st-gcn.
python tools/run.py configs/st_gcn/test.yaml --checkpoint $CHECKPOINT_PATH

# show help message
python <config file> -h

```

