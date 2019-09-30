## Skeleton-based Dataset Processing

ST-GCN was evaluated on two skeleton-based action recognition datasets: **Kinetics-skeleton** and **NTU RGB+D**.
The raw data should be converted to the proper format before training and test as below steps. Or you can download
the processed data directly from [GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb).

#### Kinetics-skeleton
[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) is a video-based dataset for action recognition which only provide raw video clips without skeleton data. Kinetics dataset include To obtain the joint locations, we first resized all videos to the resolution of 340x256 and converted the frame rate to 30 fps.  Then, we extracted skeletons from each frame in Kinetics by [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The extracted skeleton data we called **Kinetics-skeleton**(7.5GB) can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ) or [BaiduYun](https://pan.baidu.com/s/1dwKG2TLvG-R1qeIiE4MjeA#list/path=%2FShare%2FAAAI18%2Fkinetics-skeleton&parentPath=%2FShare).

After uncompressing, build the database for mmskeleton by this command:
```
python tools/data_processing/kinetics_gendata.py --data_path <path to kinetics-skeleton>
```

#### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
Only the **3D skeletons**(5.8GB) modality is required in our experiments. After that, this command should be used to build the database for training or evaluation on mmskeleton:
```
python tools/data_processing/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` is the directory path of 3D skeletons annotations of the NTU RGB+D dataset you download.
