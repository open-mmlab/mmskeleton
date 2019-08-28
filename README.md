# MMSkeleton

## Updates
[2019-08-29] MMSkeleton v0.5 is released.

## Introduction

MMSkeleton is an open source toolbox for skeleton-based human understanding.
It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

<p align="center">
    <img src="demo/recognition/demo_video.gif", width="700">
</p>

## Features

- **High extensibility**

    MMSkeleton provides a flexible framework for organizing codes and projects systematically, with the ability to extend to various tasks and scale up to complex deep models.

- **Multiple tasks**

    MMSkeleton addresses to multiple tasks in human understanding, including but not limited to:
    - skeleton-based action recognition
    - skeleton-based action generation
    - 2D/3D pose estimation
    - pose tracking


## Getting Started

First, install mmsksleton by:
``` shell
python setup.py develop
```
Any application in mmskeleton, such as training a action recognizer, is described by a configuration file. It can be started by a uniform command:
``` shell
python run.py $CONFIG_FILE [--options $OPTHION]
```
Optional arguments `options` are defined in configuration files,
check them via:
``` shell
python run.py $CONFIG_FILE -h
```

## License
The project is release under the [Apache 2.0 license](https://github.com/open-mmlab/mmskeleton/blob/master/LICENSE).

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{stgcn2018aaai,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}
```

## Contact
For any question, feel free to contact
```
Sijie Yan     : ys016@ie.cuhk.edu.hk
Yuanjun Xiong : bitxiong@gmail.com
```
