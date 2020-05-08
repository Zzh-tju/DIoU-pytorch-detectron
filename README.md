<img src="CIoU.png" width="800px"/>

## Complete-IoU Loss and Cluster-NMS for improving Object Detection and Instance Segmentation. 

This is the code for our papers:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/2005.03572)

```
@Inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}

@Article{zheng2020ciou,
  author= {Zhaohui Zheng, Ping Wang, Dongwei Ren, Wei Liu, Rongguang Ye, Qinghua Hu, Wangmeng Zuo},
  title={Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation},
  journal={arXiv:2005.03572},
  year={2020}
}
```

## Distance-IoU Loss into other SOTA detection methods can be found [here](https://github.com/Zzh-tju/DIoU). 

# Faster/Mask R-CNN with DIoU and CIoU losses implemented in - PyTorch-Detectron

## Modifications in this repository

This repository is a fork of [generalized-iou/Detectron.pytorch](https://github.com/generalized-iou/Detectron.pytorch) and [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch), with an implementation of IoU, GIoU, DIoU and CIoU losses while keeping the code as close to the original as possible. It is also possible to train the network with SmoothL1 loss as in the original code. See the options below.

### Losses

The loss can be chosen with the `MODEL.LOSS_TYPE` option in the configuration file. The valid options are currently: `[iou|giou|diou|ciou|sl1]`. At this moment, we apply bounding box loss only on final bounding box refinement layer, just as in the paper.

```
MODEL:
  LOSS_TYPE: 'diou'
```

Please take a look at `compute_iou` function of [lib/utils/net.py](lib/utils/net.py) for our DIoU and CIoU loss implementation in PyTorch.

### Normalizers

We also implement a normalizer of final bounding box refinement loss. This can be specified with the `MODEL.LOSS_BBOX_WEIGHT` parameter in the configuration file. The default value is `1.0`. We use `MODEL.LOSS_BBOX_WEIGHT` of `12.` for the four experiments.

```
MODEL:
  LOSS_BBOX_WEIGHT: 12.
```
Of course, as we observed that for dense anchor algorithms, increasing the `LOSS_BBOX_WEIGHT` appropriately can improve the performance, and the same argument is obtained in GHM (AAAI 2019)and Libra R-CNN (CVPR 2019). So, if you want to get a higher AP, just increasing it. But this may also cause unstable training, because this is equivalent to increase the learning rate.

### DIoU-NMS
NMS can be chosen with the `TEST.DIOU_NMS` option in the `lib/core/config.py` file. If set it to `False`, it means using greedy-NMS.
Besides that, we also found that for Faster R-CNN, we introduce beta1 for DIoU-NMS, that is DIoU = IoU - R_DIoU ^ {beta1}. With this operation, DIoU-NMS can perform better than default `beta1=1.0`. In our constrained search, the following values appear to work well for the DIoU-NMS in Faster R-CNN. Of course, the default `beta1=1.0` is good enough.
```
TEST.DIOU_NMS.BETA1=0.9
```

### Network Configurations

We add sample configuration files used for our experiment in `config/baselines`. Our experiments in the paper are based on `e2e_faster_rcnn_R-50-FPN_1x.yaml` as following:

```
e2e_faster_rcnn_R-50-FPN_diou_1x.yaml  # Faster R-CNN + DIoU loss
e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml   # Faster R-CNN + CIoU loss
```

### Getting Started
```
git clone https://github.com/Zzh-tju/DIoU-pytorch-detectron.git
```

### Requirements

Tested under python3.

- python packages
  - pytorch>=0.3.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.
- **NOTICE**: different versions of Pytorch package have different memory usages.

### Compilation

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```
### Data Preparation

Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download).

  And make sure to put the files as the following structure:
  ```
  coco
  ├── annotations
  |   ├── instances_minival2014.json
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   ├── instances_valminusminival2014.json
  │   ├── ...
  |
  └── images
      ├── train2014
      ├── train2017
      ├── val2014
      ├──val2017
      ├── ...
  ```
  Download coco mini annotations from [here](https://s3-us-west-2.amazonaws.com/detectron/coco/coco_annotations_minival.tgz).
  Please note that minival is exactly equivalent to the recently defined 2017 val set. Similarly, the union of valminusminival and the 2014 train is exactly equivalent to the 2017 train set.

   Feel free to put the dataset at any place you want, and then soft link the dataset under the `data/` folder:

   ```
   ln -s path/to/coco data/coco
   ```

  Recommend to put the images on a SSD for possible better training performance
  
## Train and evaluation commands

For detailed installation instruction and network training options, please take a look at the README file or issue of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Following is a sample command we used for training and testing Faster R-CNN with DIoU and CIoU.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml --use_tfboard
python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml --load_ckpt {full_path_of_the_trained_weight}
```

The following example is the evaluation of our CIoU loss:
```
INFO json_dataset_evaluator.py: 232: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
INFO json_dataset_evaluator.py: 199: Wrote json eval results to: test/detection_results.pkl
INFO task_evaluation.py:  61: Evaluating bounding boxes is done!
INFO task_evaluation.py: 180: copypaste: Dataset: coco_2017_val
INFO task_evaluation.py: 182: copypaste: Task: box
INFO task_evaluation.py: 185: copypaste: AP,AP50,AP75,APs,APm,APl
INFO task_evaluation.py: 186: copypaste: 0.3865,0.5856,0.4196,0.2132,0.4183,0.5151
```

If you want to resume training from a specific iteration's weight file, please run:
```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml --resume --use_tfboard --load_ckpt {full_path_of_the_trained_weight}
```

## Pretrained weights

Here are the trained models using the configurations in this repository.

 - [Faster RCNN + IoU](https://pan.baidu.com/s/1UGMQ90omy2MuNKbiVWbPYQ)
 - [Faster RCNN + GIoU](https://pan.baidu.com/s/1x3N7eYnylTO41klUQTlszw)
 - [Faster RCNN + DIoU](https://pan.baidu.com/s/1DtLwpSpbfNbzQ8nlHdt9Xg)
 - [Faster RCNN + CIoU](https://pan.baidu.com/s/1gUUByFBeL1DgLvHvogMUfw)
