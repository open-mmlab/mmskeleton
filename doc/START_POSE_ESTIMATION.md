## Start Pose Estimation

We provide a demo for video-based pose estimation:
```shell
mmskl pose_demo [--video $VIDEO_PATH] [--gpus $GPUS] [--$MORE_OPTIONS]
```
This demo predict pose sequences via sequentially feeding frames into the image-based human detector and the pose estimator. By default, they are  **cascade-rcnn** [1] and **hrnet** [2] respectively.
We test our demo on 8 gpus of TITAN X and get a realtime speed (27.1fps). To check the full usage, please run `mmskl pose_demo -h`. You can also refer to [pose_demo.yaml](../configs/pose_estimation/hrnet/pose_demo.yaml) for detailed configurations. 

We also provide another demo `pose_demo_HD` with a slower but more powerful detector **htc** [3]. Similarly, run:
```shell
mmskl pose_demo_HD [--video $VIDEO_PATH] [--gpus $GPUS] [--$MORE_OPTIONS]
```


### High-level APIs for testing images

Here is an example of building the pose estimator and test given images.
```python 
import mmcv
from mmskeleton.apis import init_pose_estimator, inference_pose_estimator

cfg = mmcv.Config.fromfile('configs/apis/pose_estimator.cascade_rcnn+hrnet.yaml')
video = mmcv.VideoReader('resource/data_example/skateboarding.mp4')

model = init_pose_estimator(**cfg, device=0)
for i, frame in enumerate(video):
  result = inference_pose_estimator(model, frame)
  print('Process the frame {}'.format(i))
  
  # process the result here

```

### Training and Test a Pose Estimator
Comming soon...



### Reference
```
@inproceedings{cai2018cascade,
  title={Cascade r-cnn: Delving into high quality object detection},
  author={Cai, Zhaowei and Vasconcelos, Nuno},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6154--6162},
  year={2018}
}

@article{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  journal={arXiv preprint arXiv:1902.09212},
  year={2019}
}

@inproceedings{chen2019hybrid,
  title={Hybrid task cascade for instance segmentation},
  author={Chen, Kai and Pang, Jiangmiao and Wang, Jiaqi and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Shi, Jianping and Ouyang, Wanli and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4974--4983},
  year={2019}
}
```