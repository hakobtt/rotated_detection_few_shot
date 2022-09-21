## Redet SSD

(Writing several lines here in order to help to get it working)

(The word dirty is quite soft for this repo so be ready -_-)

### Install

- install local libs

```shell
pip install -e .
```

- install pytorch. I recommend using version 1.11.0

- install mmdetection in the repo
```shell
cd mmdetection
pip install -e .
```

- install mmcv-full.
I recommend using pytorch 1.11.0 and installing mmcv-full with openmim so you don't have to wait for it to be built
```shell
pip install openmim
mim install mmcv-full==1.4.8 
```

(Sorry for not including those in setup.py)

You might still need to install several other dependencies.


---
### Train and Test



To run training
```shell
python3 ssod/tools/train.py ./configs/fair1m/soft_teacher/faster_rcnn_obb_r50_fpn_1x_fair1m_few_shot.py 
```

To run Testing
```shell
python3 ssod/tools/test.py ./configs/fair1m/soft_teacher/faster_rcnn_obb_r50_fpn_1x_fair1m_few_shot.py your/model/path 
```
