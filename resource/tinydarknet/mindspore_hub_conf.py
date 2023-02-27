
"""hub config."""
from download import download

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Model
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from model import TinyDarkNet


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss


class TinyDarkNetImageNet:
    """TinyDarkNet infer by using ImageNet data."""
    def __init__(self,
                model_name: str = "tinydarknet",
                pretrained: bool = False,
                num_classes: int = 1000,
                in_channel: int = 3,
                label_smooth_factor: float = 0.1):

        self.network = TinyDarkNet(in_channel, num_classes) if model_name == "tinydarknet" else None
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

    def infer(self, x: Tensor) -> Tensor:
        return self.model.predict(x)


if __name__ == "__main__":
    import os
    import numpy as np

    from preprocess import create_dataset_imagenet
    from postprocess import index2label

    data_path = "./data/"
    data_infer_path = os.path.join(data_path, "infer")
    print("data_path", data_infer_path)
    dataset = create_dataset_imagenet(data_infer_path)
    print("Create Dataset Sucessfully!", "Dataset Size:", dataset.get_batch_size())
    model = TinyDarkNetImageNet(pretrained=True)
    print("Create Model Sucessfully!")

    for i, image in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        image = image["image"]
        image = Tensor(image)
        prob = model.infer(image)
        label = np.argmax(prob.asnumpy(), axis=1)

        # `data_path`路径下要有`ILSVRC2012_devkit_t12`文件夹
        mapping = index2label(data_path)
        output = {int(label): mapping[int(label)]}
        print("output:", output)
