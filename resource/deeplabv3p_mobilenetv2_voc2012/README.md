# 目录

- [目录](#目录)
- [DeepLabV3+描述](##DeepLabV3+描述])
- [模型架构](##模型架构)
- [数据集](##数据集)
- [环境要求](##环境要求)
- [快速入门](##快速入门)

## [DeepLabV3+描述](#目录)

DeepLab是一系列图像语义分割模型，DeepLabv3+通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，
其骨干网络使用了MobileNetV2模型，提高了语义分割的健壮性和运行速率。

有关网络详细信息，请参阅：[DeepLabV3+](https://arxiv.org/abs/1802.02611)
`Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.`

## [模型架构](#目录)

以MobileNetV2为骨干，通过encoder-decoder进行多尺度信息的融合，使用空洞卷积进行密集特征提取。

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
./deeplabv3p_mobilenetv2_voc2012
├── data
│    └── infer_example.JPEG
├── outputs
│    └── person_infer_example.JPEG
├── hub_conf.py
├── mobilenetv2.py
├── model.py
├── preprocess.py
└── README.md
```

### 推理示例

- 加载`deeplabv3p_mobilenetv2_voc2012`模型并使用`infer`接口进行推理。

```python
import mindhub as hub

net = hub.Model("deeplabv3p_mobilenetv2_voc2012", pretrained=True)
net.infer(data_path="./data/", output_path="./outputs/")
```

```text
Matching local models: []
Matching remote models: ['deeplabv3p_mobilenetv2_voc2012']
deeplabv3p_mobilenetv2_voc2012 is not installed!
95232B [00:00, 698863.53B/s]
5120B [00:00, ?B/s]
4096B [00:00, ?B/s]
7168B [00:00, ?B/s]
13312B [00:00, 326550.45B/s]
3072B [00:00, ?B/s]
Downloading data from https://raw.githubusercontent.com/xiuyu0000/mindhub/main/resource/weight_files/deeplab_mobilenetv2.ckpt (22.3 MB)

file_sizes: 100%|██████████████████████████| 23.4M/23.4M [00:21<00:00, 1.10MB/s]
Successfully downloaded file to ./deeplab_mobilenetv2.ckpt   
infer_example.JPEG has been segmented to classes: ['person'].
```

![infer_example](./outputs/person_infer_example.JPEG)
