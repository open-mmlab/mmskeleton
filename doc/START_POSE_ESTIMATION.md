## Start Pose Estimation

We provide a demo for video-based pose estimation:
```shell
mmskl pose_demo [--video $VIDEO_PATH] [--gpus $GPUS] [--$MORE_OPTIONS]
```
This demo predict pose sequences via sequentially feeding frames into the image-based human detector and the pose estimator. By default, we use **cascade-rcnn** for detection and **hrnet** for estimation. To check the full usage, please run `mmskl pose_demo -h`. You can also refer to [pose_demo.yaml](../configs/pose_estimation/hrnet/pose_demo.yaml) for detailed configurations.

We also provide another demo `pose_demo_HD` with a slower but more powerful detector. Similarly, run:
```shell
mmskl pose_demo_HD [--video $VIDEO_PATH] [--gpus $GPUS] [--$MORE_OPTIONS]
```