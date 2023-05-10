"""hub config."""
import os
from typing import List, Dict, Optional, Callable

import numpy as np
from download import download

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Model
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase

from mindhub.models.registry import register_model
from mindhub.utils.path import load_json_file

from model import TinyDarkNet
from preprocess import create_dataset_imagenet_infer


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss


@register_model(model_name="tinydarknet_imagenet",
                model_type="image/classification",
                paper="https://github.com/pjreddie/darknet",
                pretrained=True)
class TinyDarkNetImageNet:
    """TinyDarkNet trained by using ImageNet."""
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
