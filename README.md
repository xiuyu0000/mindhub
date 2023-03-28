
<div align="center">

![](https://raw.githubusercontent.com/xiuyu0000/mindhub/main/images/mindhub_logo.gif)


[![license](https://img.shields.io/github/license/xiuyu0000/mindhub.svg)](https://github.com/xiuyu0000/mindhub/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](README_EN.md) | 中文

[简介](##MindHub简介) |
[安装](#安装) |
[快速入门](#快速入门) |
[教程](#教程) |
[模型列表](#模型列表) |

</div>

## MindHub简介

`MindHub`是一个`MindSpore`生态的模型应用工具，致力于为用户提供方便快捷的模型加载和推理功能。MindHub 的主要特点如下：

- **快速加载**：用户使用`MindHub`可以快速加载`MindSpore`的预训练模型，而无需自己下载和处理模型参数。只需几行代码，即可轻松加载模型并进行推理。
- **简单易用**：`MindHub` 提供了简单易用的 Python API，使用户可以在自己的 Python 代码中轻松使用预训练模型进行推理。API 的使用方法简单明了， 具有较好的可读性和易用性。

## 安装

### 依赖

- tqdm
- requests
- Pillow
- download
- numpy>=1.17.0
- mindspore>=2.0.0


```shell
pip install -r requirements.txt
```

用户可遵从[官方指导](https://www.mindspore.cn/install) 并根据自身使用的硬件平台选择最适合您的MindSpore版本来进行安装。如果需要在分布式条件下使用，还需安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/) 。

之后的说明将默认用户已正确安装好相关依赖。

### 源码安装

Git上最新的MindCV可以通过以下指令安装。

```shell
pip install git+https://github.com/xiuyu0000/mindhub.git
```

## 快速入门

在开始上手MindHub前，可以阅读MindHub的[自定义模型教程](tutorials/custom_model.md)，该教程可以帮助用户快速了解MindHub的各个重要组件以及
模型各项功能的使用。

以下是一些供您快速体验的代码样例。

### 模型搜索

MindHub提供了`list_models`接口，通过输出的可能的模型名称模糊匹配本地和远程仓库中的模型名称，返回可能的模型名称的列表，供用户做进一步判断。

```python
import mindhub
# 列出满足条件的预训练模型名称
mindhub.models.list_models("tinydarknet")
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
mindhub.models.Model.install_model("tinydarknet_imagenet")
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
print(mindhub.models.local_models_info("tinydarknet_imagenet"))
```

```text
{'model_name': 'tinydarknet_imagenet', 'pretrained': True, 'paper': '', 'model_type': 'image/classification'}
```

示例中由于`tinydarknet`在模型搜索中发现，所需模型未存在于本地模型注册表中，但存在于远程仓库。所以可以通过`install_model`来进行安装。安装后，
`tinydarknet_imagenet`的详细信息注册在了本地模型信息表中，可以通过`local_models_info`接口来进行查看。

### 模型移除

对于已经安装的不需要的模型，MindHub提供了`remove_model`接口，在本地模型注册表中删除该模型，并删除对应的模型文件。

```python
mindhub.models.Model.remove_model("tinydarknet_imagenet")
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
net = mindhub.models.Model(model_name="tinydarknet_imagenet", 
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
net = mindhub.models.Model(model_name="tinydarknet_imagenet", pretrained=True)
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

```python
from mindhub.utils import DownLoad
import os

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
root_dir = "./"

if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
    DownLoad().download_and_extract_archive(dataset_url, root_dir)

print(net.infer(data_path="./data/Canidae/val/dogs/", json_path="/tinydarknet_imagenet/label_map.json"))
```

```text
Data Path: ~/data/Canidae/val/dogs/
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

<details open>
<summary> 支持模型 </summary>

<details open>
<summary> 图像分类 </summary>

- TinyDarkNet - https://pjreddie.com/darknet/tiny-darknet/

</details>

</details>

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
