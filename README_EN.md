
<div align="center">

![mindhub_logo](https://raw.githubusercontent.com/xiuyu0000/mindhub/main/images/mindhub_logo.gif)


[![license](https://img.shields.io/github/license/xiuyu0000/mindhub.svg)](https://github.com/xiuyu0000/mindhub/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

English | [Chinese](README.md)

[MindHub Introduction](##MindHub Introduction) |
[Installation](##Installation) |
[Get Started](##Quick Start) |
[Tutorials](##Tutorials) |
[Model List](##Model List)

</div>

## MindHub Introduction

`MindHub` is a model application tool in the `MindSpore` ecosystem, which is committed to providing users with convenient and fast model loading and inference functions. The main features of MindHub are as follows:

- **Fast Loading**: Users can quickly load pre-trained models of `MindSpore` using `MindHub`, without the need to download and process model parameters themselves. With just a few lines of code, you can easily load the model and perform inference.
- **Easy to Use**: `MindHub` provides a simple and easy-to-use Python API, allowing users to easily use pre-trained models for inference in their own Python code. The API usage method is simple and clear, with good readability and ease of use.

## Installation

### Dependencies

- tqdm
- requests
- Pillow
- download
- numpy>=1.17.0
- mindspore>=2.0.0

```shell
pip install -r requirements.txt
```

Users can follow the [official guidance](https://www.mindspore.cn/install) and choose the most suitable version of MindSpore for your hardware platform. If you need to use it under distributed conditions, you also need to install [openmpi](https://www.open-mpi.org/software/ompi/v4.0/).

The following instructions assume that the user has correctly installed the relevant dependencies.

### Installation from Source

The latest MindCV on GitHub can be installed using the following command.

```shell
pip install https://github.com/xiuyu0000/mindhub/releases/download/v0.0.1/mindhub-0.0.1-py3-none-any.whl
```

## Get Started

Before getting started with MindHub, you can read the [Custom Model Tutorial](tutorials/custom_model.md) of MindHub, which can help users quickly understand the various important components of MindHub and the use of various model functions.

Here are some code examples for you to quickly experience.

### Model Search

MindHub provides the `list_models` interface to search for models. By outputting the possible model names that match the model name, it returns a list of possible model names in the local and remote repositories for users to make further judgments.

```python
import mindhub as hub
# List the pre-trained model names that meet the conditions
hub.list_models("tinydarknet")
```

```text
Matching local models: []
Matching remote models: ['tinydarknet_imagenet']
([], ['tinydarknet_imagenet'])
```

In the example, `tinydarknet` searched does not match locally, but matches `tinydarknet_imagenet` in the remote repository.

### Model Installation

MindHub provides the `install_model` interface to install the required model.

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

At the same time, MindHub also provides the `local_models_info` interface, which can further view the detailed information of local models.

```python
print(hub.local_models_info("tinydarknet_imagenet"))
```

```text
{'model_name': 'tinydarknet_imagenet', 'pretrained': True, 'paper': '', 'model_type': 'image/classification'}
```

In the example, since `tinydarknet` was discovered during model search, the required model does not exist in the local model registry, but exists in the remote repository. Therefore, it can be installed through `install_model`. After installation, the detailed information of `tinydarknet_imagenet` is registered in the local model information table, and can be viewed through the `local_models_info` interface.

### Model Removal

For unwanted models that have been installed, MindHub provides the `remove_model` interface to delete the model from the local model registry and delete the corresponding model file.

```python
hub.Model.remove_model("tinydarknet_imagenet")
```

```text
~/.mindhub/tinydarknet_imagenet has been successfully deleted.
```

In the example, we deleted the `tinydarknet_imagenet` model that was just installed.

### Model Loading

MindHub provides a `Model` class that allows loading the required models by either providing the local model file path and model name or just the model name.

1.  Loading Local Model Files

To load a local model file, the user needs to input the required model name, the directory path of the model file (For more details on the model file, please refer to the [Custom Model Tutorial](tutorials/custom_model.md)) and whether to download the pre-trained model.

```python
net = hub.Model(model_name="tinydarknet_imagenet", 
                directory="{YOUR_PATH}/tinydarknet_imagenet/", pretrained=True)
```

```text
Downloading data from https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt (10.4 MB)

file_sizes: 100%|██████████████████████████| 10.9M/10.9M [00:00<00:00, 22.3MB/s]
Successfully downloaded file to ./tinydarknet_ascend_v190_imagenet2012_official_cv_top1acc59.0_top5acc81.84.ckpt
```

In the above example, the `tinydarknet_imagenet` model is loaded from the `./tinydarknet_imagenet/` directory, and the pre-trained model is also downloaded and loaded.

2.  Loading Model by Name

To load a model by name, the user only needs to input the model name and whether to download the pre-trained model.

```python
net = hub.Model(model_name="tinydarknet_imagenet", pretrained=True)
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

In the above example, the model search matches that the model is not registered in the local model registry. However, the model is present in the remote repository, so it is installed first, and then the model is loaded with the downloaded pre-trained weights.

> Note: If the model name is already registered in the local model registry, it can be loaded directly without any further action. If the model name is not in the local model registry, it needs to be confirmed whether it exists in the remote repository. If it exists, it needs to be installed first, and if it does not exist, it will directly report an error.

### Model Inference

Inference is the fundamental requirement of the models in MindHub and is performed according to the method.

```python
import os

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
root_dir = "./"

if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
    hub.DownLoad().download_and_extract_archive(dataset_url, root_dir)

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

For the model in the example to perform inference, you only need to input the path where the image is located and the path where the label corresponds to where the label is located, and the inference result can be input normally.

## Tutorial

We provide relevant tutorials to help users learn how to use and contribute to MindHub.

- [Custom Model Tutorial](tutorials/custom_model.md)

## Model List

Currently, MindHub supports the following models.

<details open> 
<summary> Supported Models </summary>

<details open> 
<summary> Image Classification </summary>

- TinyDarkNet - [https://pjreddie.com/darknet/tiny-darknet](https://pjreddie.com/darknet/tiny-darknet/)

</details> </details>

## Contribution

Developers are welcome to raise issues or submit code PRs, or contribute more algorithms and models to make MindHub better.

For contribution guidelines, please refer to [CONTRIBUTING](CONTRIBUTING.md). Please follow the rules specified in the [Custom Model Tutorial](tutorials/custom_model.md) to contribute to the model interface.

## License

[Apache License 2.0](LICENSE)

## Citation

If you find MindHub helpful for your project, please consider citing:

```latex
@misc{MindSpore-Lab Hub 2023,
    title={{MindSpore-Lab Hub}:MindSpore_Lab Models Hub},
    author={MindSpore-Lab Hub Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindhub/}},
    year={2023}
}
```
