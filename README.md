# DIoU-pytorch-detectron
Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (AAAI 2020)

# Faster/Mask R-CNN with DIoU and CIoU losses implemented in - PyTorch-Detectron

If you use this work, please consider citing:

```
@article{Zhaohui_Zheng_2020_AAAI,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
  month     = {February},
  year      = {2020},
}
```

## Modifications in this repository

This repository is a fork of [generalized-iou/Detectron.pytorch](https://github.com/generalized-iou/Detectron.pytorch), with an implementation of IoU, GIoU, DIoU and CIoU losses while keeping the code as close to the original as possible. It is also possible to train the network with SmoothL1 loss as in the original code. See the options below.

### Losses

The loss can be chosen with the `MODEL.LOSS_TYPE` option in the configuration file. The valid options are currently: `[iou|giou|sl1]`. At this moment, we apply bounding box loss only on final bounding box refinement layer, just as in the paper.

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
Of course, as we observed that for dense anchor algorithms, increasing the `LOSS_BBOX_WEIGHT` appropriately can improve the performance, and the same argument is obtained in GHM (AAAI 2019)and Libra R-CNN (CVPR 2019). So, if you want to get a higher AP, just increasing it. But this may also cause unstable training, because this is equivalent to increasing the learning rate.

### Network Configurations

We add sample configuration files used for our experiment in `config/baselines`. Our experiments in the paper are based on `e2e_faster_rcnn_R-50-FPN_1x.yaml` and `e2e_mask_rcnn_R-50-FPN_1x.yaml` as following:

```
e2e_faster_rcnn_R-50-FPN_diou_1x.yaml  # Faster R-CNN + DIoU loss
e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml   # Faster R-CNN + CIoU loss
e2e_mask_rcnn_R-50-FPN_diou_1x.yaml    # Mask R-CNN + DIoU loss
e2e_mask_rcnn_R-50-FPN_ciou_1x.yaml     # Mask R-CNN + CIoU loss
```

##DIoU-NMS
NMS can be chosen with the `TEST.DIOUNMS` option in the `lib/core/config.py` file. If set it to `False`, it means using greedy-NMS.
Besides that, we also found that for Faster R-CNN, we introduce beta1 for DIoU-NMS, that is DIoU = IoU - R_DIoU ^ {beta1}. With this operation, DIoU-NMS can perform better than default beta1=1.0.
```
TEST.DIOU_NMS.BETA1=1.1
```

## Train and evaluation commands

For detailed installation instruction and network training options, please take a look at the README file or issue of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Following is a sample command we used for training and testing Faster R-CNN with DIoU and CIoU.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml --use_tfboard
python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml --load_ckpt {full_path_of_the_trained_weight}
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
 - [Faster RCNN + GIoU](https://pan.baidu.com/s/1gUUByFBeL1DgLvHvogMUfw)
