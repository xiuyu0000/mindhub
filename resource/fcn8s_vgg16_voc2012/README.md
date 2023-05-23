# 目录

- [目录](#目录)
- [FCN8s描述](##FCN8s描述])
- [模型架构](##模型架构)
- [数据集](##数据集)
- [环境要求](##环境要求)
- [快速入门](##快速入门)

## [YoloV3描述](#目录)

FCN主要用用于图像分割领域，是一种端到端的分割方法。FCN丢弃了全连接层，使得其能够处理任意大小的图像，且减少了模型的参数量，提高了模型的分割速度。FCN在编码部分使用了VGG的结构，在解码部分中使用反卷积/上采样操作恢复图像的分辨率。FCN-8s最后使用8倍的反卷积/上采样操作将输出分割图恢复到与输入图像相同大小。详情见论文。

[论文](https://arxiv.org/abs/1411.4038):  Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

## [模型架构](#目录)

FCN-8s使用丢弃全连接操作的VGG16作为编码部分，并分别融合VGG16中第3,4,5个池化层特征，最后使用stride=8的反卷积获得分割图像。

## [数据集](#目录)

Pascal VOC数据集
  - 下载链接：[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  - 数据集大小：2G
  - 图片数量：17125张
  - 物体类别数：20个
  - 训练集：12000张图像，大小为1.4G
  - 验证集：5000张图像，大小为600M
  - 标注：训练和验证标注共计241M
  - 数据格式：RGB格式图片

> 注意：数据集将会在使用前经过`preprocess.py`中的函数进行处理。

## [环境要求](#目录)

- 硬件
    - 请准备具有GPU/CPU的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多的信息请访问以下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速入门](#目录)

根据官方网站成功安装MindSpore以后，可以按照以下步骤进行训练和测试模型：

### [脚本及样例代码](#目录)

```text
./fcn8s_vgg16_voc2012
├── data
│    └── infer_example.JPEG
├── outputs
│    └── person_infer_example.JPEG
├── hub_conf.py
├── model.py
├── preprocess.py
└── README.md
```

### 推理示例

- 加载`fcn8s_vgg16_voc2012`模型并使用`infer`接口进行推理。

```python
import mindhub as hub

net = hub.Model("fcn8s_vgg16_voc2012", pretrained=True)
net.infer(data_path="./data/", output_path="./outputs/")
```

```text
Matching local models: []
Matching remote models: ['fcn8s_vgg16_voc2012']
fcn8s_vgg16_voc2012 is not installed!
52224B [00:00, 1214000.54B/s]
5120B [00:00, ?B/s]
10240B [00:00, 13276560.42B/s]
17408B [00:00, 109184.40B/s]
5120B [00:00, ?B/s]
Downloading data from https://download.mindspore.cn/models/r1.9/fcn8s_ascend_v190_voc2012_official_cv_meanIoU62.7.ckpt (1.00 GB)
file_sizes: 100%|███████████████████████████| 1.08G/1.08G [25:16<00:00, 710kB/s]
Successfully downloaded file to ./fcn8s_ascend_v190_voc2012_official_cv_meanIoU62.7.ckpt       
infer_example.JPEG has been segmented to classes: ['person'].
```

![infer_example](./outputs/person_infer_example.JPEG)
