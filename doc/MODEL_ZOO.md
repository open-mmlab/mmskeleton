## MODEL ZOO
<!-- Coming soon... -->

<!-- We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun. 
You can replace https://s3.ap-northeast-2.amazonaws.com/open-mmlab with https://open-mmlab.oss-cn-beijing.aliyuncs.com in model urls. -->

MMSkeleton usually automatically download necessary models from AWS in the runtime.

We also maintain a mirror backup on [GoogleDrive](https://drive.google.com/open?id=1zC9ptIQTUoT7RvRM9Ec651cF5Xe7pty0)
and [BaiduYun](https://pan.baidu.com/s/1iqOoQmIywuDQckgmehQ8HQ).
As a plan B, you can manually download models and put them into checkpoints cache folder of pytorch.
The folder defaults to ``~/.cache/torch/checkpoints`` in the Linux filesytem layout.