# Spatial Temporal Graph Convolutional Networks (ST-GCN)
A graph convolutional network for skeleton based action recognition.

## Introduction
This repository holds the codebase, dataset and models for the paper

**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018.

[Arxiv Preprint]

## Usage Guide
### Prerequisites
There are a few dependencies to run the code. The major libraries we used are
- [PyTorch](http://pytorch.org/)
- NumPy

### Data Preparation
We experimented on two action recognition datasts: [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) and [Kinetics-skeleton](https://s3-us-west-1.amazonaws.com/yysijie-data/kinetics-skeleton.zip). 
##### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). Only the **3D skeletons**(5.8GB) modality is required in our experiments. After that, ```tools/ntu_gendata.py``` should be used to build the database for training or evaluation:
```
tools/ntu_gendata.py --data_path <path to nturgbd>
```
where the ```<path to nturgbd>``` points to the 3D skeletons modality of NTU RGB+D dataset you download, for example ```data/NTU-RGB-D/nturgbd+d_skeletons```.
##### Kinetics-skeleton
[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) is a video-based dataset for action recognition which only provide raw video clips without skeleton data. To obatin the joint locations, we first resized all videos to the resolution of 340x256 and converted the frame rate to 30 fps.  Then, we extracted skeletons from each frame in Kinetics by [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The extracted skeleton data we called **Kinetics-skeleton**(7.5GB) can be directly downloaded from [here](https://s3-us-west-1.amazonaws.com/yysijie-data/kinetics-skeleton.zip).

It is highly recommended storing data in the **SSD** rather than HDD for efficiency.


##  Testing Provided Models
##### Get trained models
We provided the trained model weithts of  **Temporal Conv** [1] and our **ST-GCN**. The model weights can be downloaded by running the script
```
bash tools/get_reference_models.sh
```

##### Evaluation
Model evaluation can be achieved by this command:
```
main.py --phase test --config <path to training config> --weights <path to model weights>
```
##### Results
| Model| Kinetics-<br>skeleton (%)|NTU RGB+D <br> Cross View (%) |NTU RGB+D <br> Cross Subject (%) |
| :------| :------: | :------: | :------: |
|[Temporal Conv](https://arxiv.org/abs/1704.04516) [1] | 20.3    | 83.1     |  74.3    |
|**ST-GCN** (Ours)| **30.7**| **88.3** | **80.5** | 

[1] Kim, T. S., and Reiter, A. 2017. Interpretable 3d human action analysis with temporal convolutional networks. In BNMW CVPRW. 

## Training
To train a new model, use the ```main.py``` script. For example: 
```
main.py --config config/Kinetics/ST-GCN.yaml
```
We have provided the necessary solver configs under the ```./config```. The training results will be saved under the ```./work_dir``` by default.

You can modify the training parameters such as ```batch-size``` and ```device``` in the command line or config files. The order of priority is:  command line > config file > default parameter. For more information, use 
```
main.py -h
```



