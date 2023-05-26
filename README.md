
<div align="center">

![mindhub_logo](https://raw.githubusercontent.com/xiuyu0000/mindhub/main/images/mindhub_logo.gif)


[![license](https://img.shields.io/github/license/xiuyu0000/mindhub.svg)](https://github.com/xiuyu0000/mindhub/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](README_EN.md) | 中文

[MindHub简介](##MindHub简介) |
[安装](##安装) |
[快速入门](##快速入门) |
[功能介绍](##功能介绍) |
[教程](##教程) |
[模型列表](##模型列表)

</div>

## MindHub简介

`MindHub`是一个`MindSpore`生态的模型应用工具，致力于为用户提供方便快捷的模型加载和推理功能。MindHub 的主要特点如下：

- **快速加载**：用户使用`MindHub`可以快速加载`MindSpore`的预训练模型，而无需自己下载和处理模型参数。只需几行代码，即可轻松加载模型并进行推理。
- **简单易用**：`MindHub` 提供了简单易用的 Python API，使用户可以在自己的 Python 代码中轻松使用预训练模型进行推理。API 的使用方法简单明了， 具有较好的可读性和易用性。

## 安装

### 依赖

- tqdm
- Pillow
- numpy>=1.17.0
- mindspore>=2.0.0


```shell
pip install -r requirements.txt
```

用户可遵从[官方指导](https://www.mindspore.cn/install) 并根据自身使用的硬件平台选择最适合您的MindSpore版本来进行安装。如果需要在分布式条件下使用，还需安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/) 。

之后的说明将默认用户已正确安装好相关依赖。

### 源码安装

Github上最新的MindCV可以通过以下指令安装。

```shell
pip install pip install https://github.com/xiuyu0000/mindhub/releases/download/v0.0.1/mindhub-0.0.1-py3-none-any.whl

```

## 快速入门

在开始上手MindHub前，可以阅读MindHub的[自定义模型教程](tutorials/custom_model.md)，该教程可以帮助用户快速了解MindHub的各个重要组件以及
模型各项功能的使用。

以下是一些供您快速体验的代码样例。

```python
import mindhub as hub

net = hub.Model("tinydarknet_imagenet", pretrained=True)
print(net.infer(data_path="./data/infer_example.JPEG", json_path="./label_map.json"))
```

```text
Matching local models: []
Matching remote models: ['tinydarknet_imagenet']
tinydarknet_imagenet is not installed!
4096B [00:00, 2369637.13B/s]           
20480B [00:00, 6247225.16B/s]           
4096B [00:00, 1779006.85B/s]           
5120B [00:00, 2905538.69B/s]           
4096B [00:00, 2582274.04B/s]           
Downloading data from https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt (10.4 MB)

file_sizes: 100%|██████████████████████████| 10.9M/10.9M [00:00<00:00, 23.1MB/s]
Successfully downloaded file to ./tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
Data Path: ./data/infer_example.JPEG
[{'275': 'African hunting dog'}]
```

![infer_example](./resource/tinydarknet_imagenet/data/infer_example.JPEG)

## 功能介绍

MindHub目前版本包括模型搜索，模型安装/移除和模型加载等功能，具体使用方法如下所示。

### 模型搜索

MindHub提供了`list_models`接口，通过输出的可能的模型名称模糊匹配本地和远程仓库中的模型名称，返回可能的模型名称的列表，供用户做进一步判断。

```python
import mindhub as hub
# 列出满足条件的预训练模型名称
hub.list_models("tinydarknet")
```

```text
Matching local models: []
Matching remote models: ['tinydarknet_imagenet']
([], ['tinydarknet_imagenet'])
```

示例中搜索的`tinydarknet`在本地没有匹配，在远程仓库匹配到了`tinydarknet_imagenet`。

### 模型安装

MindHub提供了`install_model`接口，来安装所需模型。

```python
hub.Model.install_model("tinydarknet_imagenet")
```

```text
Matching local models: []
Matching remote models: ['tinydarknet_imagenet']
tinydarknet_imagenet is not installed!
4096B [00:00, 2120447.94B/s]           
20480B [00:00, 7109107.50B/s]           
4096B [00:00, 1644006.62B/s]           
5120B [00:00, 1940438.83B/s]           
4096B [00:00, 2142128.33B/s]           
'~/.mindhub/tinydarknet_imagenet'
```

同时MindHub还提供了`local_models_info`接口，可以进一步查看本地模型的详细信息。

```python
print(hub.local_models_info("tinydarknet_imagenet"))
```

```text
{'model_name': 'tinydarknet_imagenet', 'pretrained': True, 'paper': '', 'model_type': 'image/classification'}
```

示例中由于`tinydarknet`在模型搜索中发现，所需模型未存在于本地模型注册表中，但存在于远程仓库。所以可以通过`install_model`来进行安装。安装后，
`tinydarknet_imagenet`的详细信息注册在了本地模型信息表中，可以通过`local_models_info`接口来进行查看。

### 模型移除

对于已经安装的不需要的模型，MindHub提供了`remove_model`接口，在本地模型注册表中删除该模型，并删除对应的模型文件。

```python
hub.Model.remove_model("tinydarknet_imagenet")
```

```text
~/.mindhub/tinydarknet_imagenet has been successfully deleted.
```

示例中我们删除了刚刚安装的`tinydarknet_imagenet`模型。

### 模型加载

MindHub提供了`Model`类，可以通过加载本地模型文件或者模型名称来加载所需的模型。

1. 本地模型文件加载

通过本地模型文件加载，需要同时输入所需的模型名称和模型文件所在文件夹路径（关于模型文件的详细信息，
请参阅[自定义模型教程](tutorials/custom_model.md)）以及是否需要下载预训练模型。

```python
net = hub.Model(model_name="tinydarknet_imagenet", 
                diretory="{YOUR_PATH}/tinydarknet_imagenet/", pretrained=True)
```

```text
Downloading data from https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt (10.4 MB)

file_sizes: 100%|██████████████████████████| 10.9M/10.9M [00:00<00:00, 22.3MB/s]
Successfully downloaded file to ./tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
```

示例中通过加载`./tinydarknet_imagenet/`文件夹中的模型文件加载了`tinydarknet_imagenet`模型，并加载了预训练模型。

2. 模型名称加载

通过输入模型名称加载，只需要输入模型名称以及是否需要下载预训练模型。

```python
net = hub.models.Model(model_name="tinydarknet_imagenet", pretrained=True)
```

```text
Matching local models: []
Matching remote models: ['tinydarknet_imagenet']
tinydarknet_imagenet is not installed!
4096B [00:00, 2369637.13B/s]           
20480B [00:00, 6247225.16B/s]           
4096B [00:00, 1779006.85B/s]           
5120B [00:00, 2905538.69B/s]           
4096B [00:00, 2582274.04B/s]           
Downloading data from https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt (10.4 MB)

file_sizes: 100%|██████████████████████████| 10.9M/10.9M [00:00<00:00, 23.1MB/s]
Successfully downloaded file to ./tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
```

示例中通过模型搜索发现本地模型注册表中不存在该模型，但是远程仓库中存在，因此进行模型安装，之后进行模型加载以及预训练权重的下载和加载。

> 注：如果模型名称已经注册在本地模型注册表中，不需要其他操作可以直接加载；如果模型名称未在 本地模型注册表中，则需要进一步确认是否存在于远程仓库，
> 如果存在则需要先进行，如果不存在则直接报错。

### 模型推理

推理是MindHub中的模型所要求的基本功能，根据贡献者定义的方法来进行推理。

- 推理单张图片。

```python
print(net.infer(data_path="./data/infer_example.JPEG", json_path="./label_map.json"))
```

```text
Data Path: ./data/infer_example.JPEG
[{'275': 'African hunting dog'}]
```

- 推理文件夹下所有图片。

```python
import os

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
root_dir = "./"

if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
    hub.DownLoad().download_and_extract_archive(dataset_url, root_dir)

print(net.infer(data_path="./data/Canidae/val/dogs/", json_path="./label_map.json"))
```

```text
Data Path: ./data/Canidae/val/dogs/
[{'158': 'toy terrier'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, {'161': 'basset'}, {'151': 'Chihuahua'}, 
{'263': 'Pembroke'}, {'263': 'Pembroke'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, 
{'171': 'Italian greyhound'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, {'6': 'stingray'}, 
{'253': 'basenji'}, {'151': 'Chihuahua'}, {'227': 'kelpie'}, {'151': 'Chihuahua'}, {'151': 'Chihuahua'}, 
{'151': 'Chihuahua'}, {'151': 'Chihuahua'}, {'158': 'toy terrier'}, {'151': 'Chihuahua'}, {'669': 'mosquito net'}, 
{'151': 'Chihuahua'}, {'173': 'Ibizan hound'}, {'156': 'Blenheim spaniel'}, {'237': 'miniature pinscher'}, 
{'416': 'balance beam'}]
```

示例中的模型要进行推理只需要输入图片所在的路径和标签对应表所在的路径，就可以正常输出推理结果。

## 教程
我们提供了相关教程，帮助用户学习如何使用和贡献MindHub

- [自定义模型教程](tutorials/custom_model.md)

### 模型列表

目前，MindHub支持以下模型。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>论文</th>
      <th>支持</th>
    </tr>
  </thead>
  <tbody>
    <tr>
	<td colspan=6 align="center"><b>Vision - Segmentation</b></td>
	</tr>
    <tr>
      <td>U-Net</td>
      <td><a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>DeepLabV3+</td>
      <td><a href="https://arxiv.org/abs/1802.02611">Encoder-decoder with atrous separable convolution for semantic image segmentation.</a></td>
      <td>:white_check_mark:</td>
    </tr>
    <tr>
      <td>FCN8s</td>
      <td><a href="https://arxiv.org/abs/1411.4038">Fully convolutional networks for semantic segmentation.</a></td>
      <td>:white_check_mark:</td>
    </tr>
    <tr>
      <td>Mask R-CNN</td>
      <td><a href="https://arxiv.org/abs/1703.06870v3">Mask R-CNN</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>PointNet</td>
      <td><a href="https://arxiv.org/abs/1612.00593v2">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - Classification</b></td>
	</tr>
    <tr>
      <td>tinydarknet</td>
      <td><a href="https://pjreddie.com/darknet/tiny-darknet/">Tiny Darknet</a></td>
      <td>:white_check_mark:</td>
    </tr>
    <tr>
      <td>resnet</td>
      <td><a href="https://arxiv.org/abs/1512.03385v1">Deep Residual Learning for Image Recognition</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>mobilenetv2</td>
      <td><a href="https://arxiv.org/abs/1801.04381v4">MobileNetV2: Inverted Residuals and Linear Bottlenecks</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>efficientnet</td>
      <td><a href="https://arxiv.org/abs/1905.11946v5">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>cspnet</td>
      <td><a href="https://arxiv.org/abs/1911.11929v1">CSPNet: A New Backbone that can Enhance Learning Capability of CNN</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>vit</td>
      <td><a href="https://arxiv.org/abs/2010.11929v2">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - Face Verification</b></td>
	</tr>
    <tr>
      <td>facenet</td>
      <td><a href="https://arxiv.org/abs/1503.03832v3">FaceNet: A Unified Embedding for Face Recognition and Clustering</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>arcface</td>
      <td><a href="https://arxiv.org/abs/1801.07698v4">ArcFace: Additive Angular Margin Loss for Deep Face Recognition</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>retinaface</td>
      <td><a href="https://arxiv.org/abs/1905.00641v2">RetinaFace: Single-stage Dense Face Localisation in the Wild</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - Detection</b></td>
	</tr>
    <tr>
      <td>yolov3</td>
      <td><a href="https://arxiv.org/abs/1804.02767v1">YOLOv3: An Incremental Improvement</a></td>
      <td>:white_check_mark:</td>
    </tr>
    <tr>
      <td>yolov4</td>
      <td><a href="https://arxiv.org/abs/2004.10934v1">YOLOv4: Optimal Speed and Accuracy of Object Detection</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>yolov7</td>
      <td><a href="https://arxiv.org/abs/2207.02696v1">YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>detr</td>
      <td><a href="https://arxiv.org/abs/2005.12872v3">End-to-End Object Detection with Transformers</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>ssd</td>
      <td><a href="https://arxiv.org/abs/1512.02325v5">SSD: Single Shot MultiBox Detector</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>fast r-cnn</td>
      <td><a href="https://arxiv.org/abs/1504.08083v2">Fast R-CNN</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - OCR</b></td>
	</tr>
    <tr>
      <td>DBNet</td>
      <td><a href="https://arxiv.org/abs/2010.11566v1">DBNET: DOA-driven beamforming network for end-to-end farfield sound source separation</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>CRNN</td>
      <td><a href="https://arxiv.org/abs/1706.01069">CRNN: A Joint Neural Network for Redundancy Detection</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - Video</b></td>
	</tr>
    <tr>
      <td>fairmot</td>
      <td><a href="https://arxiv.org/abs/2004.01888v6">FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>swin3d</td>
      <td><a href="https://arxiv.org/abs/2106.13230v1">Video Swin Transformer</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - 3D</b></td>
	</tr>
    <tr>
      <td>second</td>
      <td><a href="https://arxiv.org/abs/2106.13230v1">Video Swin Transformer</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>PointNet++</td>
      <td><a href="https://arxiv.org/abs/1706.02413v1">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Vision - 3D Restruction</b></td>
	</tr>
    <tr>
      <td>DeepLM</td>
      <td><a href="http://openaccess.thecvf.com//content/CVPR2021/html/Huang_DeepLM_Large-Scale_Nonlinear_Least_Squares_on_Deep_Learning_Frameworks_Using_CVPR_2021_paper.html">DeepLM: Large-Scale Nonlinear Least Squares on Deep Learning Frameworks Using Stochastic Domain Decomposition</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>DecoMR</td>
      <td><a href="https://arxiv.org/abs/2006.05734v2">3D Human Mesh Regression with Dense Correspondence</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>NLP - Machine Translation</b></td>
	</tr>
    <tr>
      <td>seq2seq</td>
      <td><a href="https://arxiv.org/abs/1409.3215v3">Sequence to Sequence Learning with Neural Networks</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>speech_transformer</td>
      <td><a href="https://ieeexplore.ieee.org/document/8682586">The Speechtransformer for Large-scale Mandarin Chinese Speech Recognition</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>NLP - Talking</b></td>
	</tr>
    <tr>
      <td>dgu</td>
      <td><a href="https://github.com/PaddlePaddle/models/tree/release/1.6/PaddleNLP/PaddleDialogue">DGU: Dialogue General Understanding</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>dam</td>
      <td><a href="https://aclanthology.org/P18-1103.pdf">Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>NLP - Language Modeling</b></td>
	</tr>
    <tr>
      <td>gpt-3</td>
      <td><a href="https://arxiv.org/abs/2005.14165v4">Language Models are Few-Shot Learners</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>pangu</td>
      <td><a href="https://arxiv.org/abs/2104.12369v1">PanGu-α
: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>bert</td>
      <td><a href="https://arxiv.org/abs/1810.04805v2">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Recommand - CTR</b></td>
	</tr>
    <tr>
      <td>Wide & Deep</td>
      <td><a href="https://arxiv.org/abs/1606.07792v1">Wide & Deep Learning for Recommender Systems</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>DeepFM</td>
      <td><a href="https://arxiv.org/abs/1804.04950v2">DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>EDCN</td>
      <td><a href="https://dl.acm.org/doi/abs/10.1145/3459637.3481915">Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Audio - Speech Recognition</b></td>
	</tr>
    <tr>
      <td>conformer</td>
      <td><a href="https://arxiv.org/abs/2005.08100v1">Conformer: Convolution-augmented Transformer for Speech Recognition</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>deepspeechv2</td>
      <td><a href="https://arxiv.org/abs/1512.02595v1">Deep Speech 2: End-to-End Speech Recognition in English and Mandarin</a></td>
      <td>:x:</td>
    </tr>
    <tr>
	<td colspan=6 align="center"><b>Generation - Image Generation</b></td>
	</tr>
    <tr>
      <td>dcgan</td>
      <td><a href="https://arxiv.org/abs/1511.06434v2">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>stable diffusion v2</td>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html">High-Resolution Image Synthesis with Latent Diffusion Models</a></td>
      <td>:x:</td>
    </tr>
    <tr>
      <td>wukong-huahua</td>
      <td><a href="https://github.com/mindspore-lab/minddiffusion/tree/main/vision/wukong-huahua">Wukong-Huahua</a></td>
      <td>:x:</td>
    </tr>
  </tbody>
</table>

## 贡献

欢迎开发者用户提issue或提交代码PR，或贡献更多的算法和模型，一起让MindHub变得更好。

有关贡献指南，请参阅[CONTRIBUTING](CONTRIBUTING.md)。请遵循[自定义模型模型教程](tutorials/custom_model.md)所规定的规则来贡献模型接口：)

## 许可证

[Apache License 2.0](LICENSE)

## 引用
如果你觉得MindHub对你的项目有帮助，请考虑引用：

```latex
@misc{MindSpore-Lab Hub 2023,
    title={{MindSpore-Lab Hub}:MindSpore_Lab Models Hub},
    author={MindSpore-Lab Hub Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindhub/}},
    year={2023}
}
```
