# 目录

- [目录](#目录)
- [Tiny-DarkNet描述](##Tiny-DarkNet描述])
- [模型架构](##模型架构)
- [数据集](##数据集)
- [环境要求](##环境要求)
- [快速入门](##快速入门)

## [Tiny-DarkNet描述](#目录)

Tiny-DarkNet是Joseph Chet Redmon等人提出的一个16层的针对于经典的图像分类数据集ImageNet所进行的图像分类网络模型。 Tiny-DarkNet作为作者为了满足用户对较小模型规模的需求而尽量降低模型的大小设计的简易版本的Darknet，具有优于AlexNet和SqueezeNet的图像分类能力，同时其只使用少于它们的模型参数。为了减少模型的规模，该Tiny-DarkNet网络没有使用全连接层，仅由卷积层、最大池化层、平均池化层组成。

更多Tiny-DarkNet详细信息可以参考[官方介绍](https://pjreddie.com/darknet/tiny-darknet/)

## [模型架构](#目录)

具体而言, Tiny-DarkNet网络由**1×1 conv**, **3×3 conv**, **2×2 max**和全局平均池化层组成，这些模块相互组成将输入的图片转换成一个**1×1000**的向量。

## [数据集](#目录)

以下将介绍模型中使用数据集以及其出处：
<!-- Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below. -->

<!-- Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)  -->

<!-- Dataset used ImageNet can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size: 125G, 1250k colorful images in 1000 classes
  - Train: 120G, 1200k images
  - Test: 5G, 50k images
- Data format: RGB images.
  - Note: Data will be processed in src/dataset.py  -->

所使用的数据集可参考[论文](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- 数据集规模：125G，1250k张分别属于1000个类的彩色图像
    - 训练集: 120G,1200k张图片
    - 测试集: 5G, 50k张图片
- 数据格式: RGB格式图片
    - 注意: 数据将会被 src/dataset.py 中的函数进行处理

## [环境要求](#目录)

- 硬件
    - 请准备具有GPU的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多的信息请访问以下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速入门](#目录)

根据官方网站成功安装MindSpore以后，可以按照以下步骤进行训练和测试模型：

### [脚本及样例代码](#目录)

```text
./tinydarknet
├── data
│   ├── ILSVRC2012_devkit_t12
│   │   ├── COPYING
│   │   ├── data
│   │   │   ├── ILSVRC2012_validation_ground_truth.txt
│   │   │   └── meta.mat
│   │   ├── evaluation
│   │   │   ├── compute_overlap.m
│   │   │   ├── demo_eval.m
│   │   │   ├── demo.val.pred.det.txt
│   │   │   ├── demo.val.pred.txt
│   │   │   ├── eval_flat.m
│   │   │   ├── eval_localization_flat.m
│   │   │   ├── get_class2node.m
│   │   │   ├── make_hash.m
│   │   │   ├── VOCreadrecxml.m
│   │   │   ├── VOCreadxml.m
│   │   │   └── VOCxml2struct.m
│   │   └── readme.txt
│   └── infer
│       └── n02090622
│           └── n02090622_8464.JPEG
├── mindspore_hub_conf.py
├── model.py
├── postprocess.py
├── preprocess.py
└── tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
```

### GPU单卡推理示例

```shell
python mindspore_hub_conf.py
```

```text
data_path: ./data/infer
Create Dataset Sucessfully! Dataset Size: 1
Downloading data from https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt (10.4 MB)

file_sizes: 100%|██████████████████████████| 10.9M/10.9M [00:00<00:00, 16.1MB/s]
Successfully downloaded file to ./tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
Create Model Sucessfully!
output: {175: 'otterhound'}
```
