# Spatial Temporal Graph Convolutional Networks (ST-GCN)
A graph convolutional network for skeleton based action recognition.

## Introduction
This repository holds the codes and models for the paper

**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018.

[Arxiv Preprint]

## Usage Guide
### Prerequisites
There are a few dependencies to run the code. The major libraries we used are
- [Pytorch](http://pytorch.org/)
- NumPy

### Data Preparation
We experimented on two action recognition datasts: [NTU-RGB-D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) and [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). 
#### NTU-RGB-D
NTU-RGB-D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). Only the **3D skeletons**(5.8GB) modality is required in our experiments. After that, ```tools/ntu_gendata.py``` should be used to build the database for training or evaluation:
```
tools/ntu_gendata.py --data_path <path to nturgbd>
```
where the ```<path to nturgbd>``` points to the 3D skeletons modality of NTU-RGB-D you download, for example ```data/NTU-RGB-D/nturgbd+d_skeletons```.
#### Kinetics
Kinetics is a video-based dataset for action recognition which only provide raw video clips without skeleton data. To obatin the joint locations, we first resized all videos to the resolution of 340x256 and converted the frame rate to 30 fps.  Then, we extracted skeletons from each frame in Kinetics by [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The **extracted skeleton data** (7.5GB) can be directly downloaded from either of the below two links:
- [Google drive](https://google.drive.com) 
- [BaiduYun](https://yun.baidu.com)
	

## Training
To train a new model, use the ```main.py``` script. For example: 
```
main.py --config config/Kinetics/ST-GCN.yaml
```
We have provided the necessary solver configs under the ```./config```. The training results will be saved under the ```./work_dir```.

You can modify the training parameters such as ```batch-size``` and ```device``` in the command line or config files. The order of priority is:  command line > config file > default parameter.



## Evaluation

Model evaluation can be achieved by this command:
```
bash evaluation.sh
```

```
main.py --phase test --config <path to training config> --weights <path to model weights>
```
