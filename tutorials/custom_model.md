# 如何创建自己的Model

MindHub是一个为开发者提供模型共享的平台，其中包含了丰富的预训练模型，开发者可以直接使用这些模型，也可以将自己的模型上传到 MindSpore Hub 上与
其他开发者分享。它向应用开发者提供了简单易用的模型加载和推理API，使得用户可以基于预训练模型进行推理。

> 注: 在开始本教程之前，你需要安装MindSpore并了解如何使用它构造模型。
> 详情请参考：https://www.mindspore.cn/tutorials/zh-CN/master/index.html

本教程以`tinydarknet_imagenet`为例，对想要将模型发布到MindHub的模型的开发者介绍了模型构造的步骤。

## 信息准备

对于开发者发布的模型，我们希望开发者可以提供模型相关的详细信息，这些信息在加载模型时会被随模型一起注册到本地模型信息表中，方便其他需要使用该模型的用户查看，以分辨该模型是否是他们实际需要使用的模型。

```text
model_name: tinydarknet_imagenet
pretrained: True
paper: 'https://github.com/pjreddie/darknet'
model_type: 'image/classification'
```

- model_name: 模型规格名称，应该遵循`{模型规格}_{数据集}`的方式命名。
- pretrained: 该规格的模型是否存在预训练模型。
- paper: 模型的来源，如果存在相关论文请填写论文的全程，如果不存在，可以填写发布该模型源码的。
- model_type: 模型的任务类型。

## 发布模型到MindHub

由于MindHub需要兼容多种不同的领域的模型，因此我们不尽量避免对构造模型文件添加太多的限制，但是为了开发者贡献的模型可以顺利地加载和进行推理，仍然需要
遵循一些规则和建议。

我们仍然以`tinydarknet_imagenet`为例，来展示如果构建模型文件。

### 必要的目录和文件

1. 参照[`tinydarknet_imagenet`示例](https://github.com/xiuyu0000/mindhub/tree/main/resource/tinydarknet_imagenet)，在`resource/`文件夹下创建`{模型规格}_{数据集}`文件夹，该文件夹的目录结构为：

   ```text
   tinydarknet_imagenet
   ├── README.md
   ├── label_map.json
   ├── model.py
   ├── preprocess.py
   └── mindspore_hub_conf.py
   ```

2. 其中`README.md`和`mindspore_hub_conf.py`两个文件是必须的。
   - `README.md`中需要对模型的基本信息进行描述，至少要包括模型的基本介绍，benchmark, 模型的安装和使用方法。
   - `mindspore_hub_conf.py`的作用是实例化模型和数据集并执行注册器，将模型注册到用户的本地模型注册表中。该文件的名称不可改变，否则将无法正常
   加载模型。模型和数据集的实例化方式没有严格的限制，如果模型较为简单可以直接在该文件中进行定义，如果模型较为复杂且包含多个规格可以参照示例中的做法，
   创建一个`model.py`文件并对定义的模型在`mindspore_hub_conf.py`中进行引用。同理数据集也参照这个原则，如果不复杂直接在`mindspore_hub_conf.py`中
   进行定义，如果复杂可以参照示例创建`preprocess.py`来定义数据集接口，并在`mindspore_hub_conf.py`中引用。
   - 另外，`label_map.json`是imagenet数据集分类序号和类别名称的对应表。
   
   > 注: MindHub推荐使用最简洁的方式来实现功能，希望开发者在贡献模型时以最少的代码做模型加载和模型推理，例如，示例中imagenet数据集的分类序号和
   > 类别名称的对应表需要较为复杂的操作才能得到，我们为了省略这部分代码就直接上传了包含处理后的结果的文件，我们同样推荐开发者做类似的做操作，除非
   > 要上传的文件过大，比如处理后的大型数据集。

### 模型文件的编写规则

1. 定义要注册的模型的类`TinyDarkNetImageNet`并将准备的模型信息输入到模型注册器`register_model`中。

```python
from mindhub import register_model

@register_model(model_name="tinydarknet_imagenet",
                model_type="image/classification",
                paper="https://github.com/pjreddie/darknet",
                pretrained=True)
class TinyDarkNetImageNet:
    """TinyDarkNet infer by using ImageNet data."""
    def __init__(self,
                model_name: str = "tinydarknet_imagenet",
                pretrained: bool = False,
                num_classes: int = 1000,
                in_channel: int = 3,
                label_smooth_factor: float = 0.1):
       ...
```

2. 执行必要的初始化，如模型的实例化，预训练权重的加载，其他模型相关模块的加载，模型的封装等。

   > 注: 开发者需要将预训练模型托管在可以访问的存储位置并提供下载地址，以便在加载时下载。

```python
class TinyDarkNetImageNet:
    """TinyDarkNet infer by using ImageNet data."""
    def __init__(self,
                model_name: str = "tinydarknet_imagenet",
                pretrained: bool = False,
                num_classes: int = 1000,
                in_channel: int = 3,
                label_smooth_factor: float = 0.1):

        self.network = TinyDarkNet(in_channel, num_classes) if model_name == "tinydarknet_imagenet" else None
        self.ckpt_url = "https://download.mindspore.cn/models/r1.9/tinydarknet_ascend_v190_imagenet2012" \
                        "_official_cv_top1acc59.0_top5acc81.84.ckpt"
        self.ckpt_path = "./tinydarknet_ascend_v190_imagenet2012_official_cv" \
                         "_top1acc59.0_top5acc81.84.ckpt"

        if pretrained:
            path = download(self.ckpt_url, self.ckpt_path, replace=True)
            param_dict = ms.load_checkpoint(path)
            ms.load_param_into_net(self.network, param_dict)

        self.loss = CrossEntropySmooth(sparse=True,
                                       reduction="mean",
                                       smooth_factor=label_smooth_factor,
                                       num_classes=num_classes)

        self.model = Model(self.network, loss_fn=self.loss)
    ...
```

3. 完善模型推理的逻辑，MindHub推荐开发者尽可能的将数据前处理，模型推理和模型后处理都加入到`infer`方法中，以便用户可以快速简洁使用模型的推理功能，
而不需要做其他非必要的操作。

```python
class TinyDarkNetImageNet:
    def infer(self,
              data_path: str,
              json_path: str,
              transform: Optional[Callable] = None,
              batch_size: int = 1,
              num_parallel_workers: int = 1,
              ) -> List[Dict]:

        if os.path.exists(data_path):
            print(f"Data Path: {data_path}")
            dataset = create_dataset_imagenet_infer(data_path, transform, batch_size, num_parallel_workers)
            mapping = load_json_file(json_path)

            outputs = []
            for image in dataset.create_dict_iterator():
                image = image["image"]
                prob = self.model.predict(image)
                label = np.argmax(prob.asnumpy(), axis=1)
                output = {str(label[0]): mapping[str(label[0])]}
                outputs.append(output)
        else:
            raise FileNotFoundError(f"Please check whether the path {data_path} exists!")

        return outputs
```

同时，为了可以帮助开发者简化代码，MindHub还提供了开发中常用的图片读取，路径解析以及下载解压等接口，来帮助开发者处理一些常见操作。

<details open> 
<summary>  常用接口 </summary>

<details open> 
<summary> 图片读取 </summary>

- [`read_dataset`]()
- [`image_read`]()

</details>

<details open> 
<summary> 路径解析  </summary>

- [`check_file_exist`]()
- [`check_dir_exist`]()
- [`save_json_file`]()
- [`load_json_file`]()
- [`detect_file_type`]()

</details> 

<details open> 
<summary> 下载解压  </summary>

- [`DownLoad.download_and_extract_archive`]()
- [`DownLoad. download_github_folder`]()

</details> 

</details>

> 完整代码请参阅[mindspore_hub_conf.py](https://github.com/xiuyu0000/mindhub/blob/main/resource/tinydarknet_imagenet/mindspore_hub_conf.py)。