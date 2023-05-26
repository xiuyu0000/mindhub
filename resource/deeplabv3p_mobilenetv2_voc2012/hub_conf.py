import os
import copy
from typing import Optional

import numpy as np
from PIL import Image
from download import download

import mindspore as ms
from mindspore import nn
from mindhub.models.registry import register_model
from mindhub.utils.images import image_read, read_dataset

from model import DeepLab
from preprocess import infer_batch_scales, cvt_color

NAME_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                "tvmonitor"]


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


@register_model(model_name="deeplabv3p_mobilenetv2_voc2012",
                model_type="image/segmentation",
                paper="https://arxiv.org/abs/1802.02611",
                pretrained=True)
class DeepLabV3PlusMobileNetV2VOC2012:
    def __init__(self,
                 model_name: str = "deeplabv3p_mobilenetv2_voc2012",
                 pretrained: bool = False,
                 weight_path: Optional[str] = None,
                 num_classes: int = 21,
                 output_stride: int = 16):

        self.net = DeepLab(num_classes, downsample_factor=output_stride) \
            if model_name == "deeplabv3p_mobilenetv2_voc2012" else None

        self.ckpt_url = "https://download.mindspore.cn/models/r1.9/deeplabv3plus_s16_ascend_v190_voc2012_" \
                        "research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt"
        if output_stride == 8:
            self.ckpt_url = "https://download.mindspore.cn/models/r1.9/deeplabv3plus_s8r2_ascend_v190_voc2012_" \
                            "research_cv_s8acc79.62_s8multiscale80.32_s8multiscaleflip80.61.ckpt"
        self.ckpt_path = f"./{os.path.basename(self.ckpt_url)}"

        if weight_path:
            param_dict = ms.load_checkpoint(weight_path)
            ignored_params = ms.load_param_into_net(self.net, param_dict)
        elif pretrained:
            path = download(self.ckpt_url, self.ckpt_path, replace=False)
            param_dict = ms.load_checkpoint(path)
            ignored_params = ms.load_param_into_net(self.net, param_dict)

        if ignored_params[0]:
            raise ValueError(f"Omitted parameters: {ignored_params[0]}.")

        self.network = BuildEvalNetwork(self.net)

    def infer(self, data_path,
              output_path,
              scales=(1,),
              base_crop_size=513,
              batch_size=1,
              flip=True,
              target_classes=None):
        self.network.set_train(False)
        if not target_classes:
            target_classes = list(range(len(NAME_CLASSES)))

        images_path = read_dataset(data_path)
        batch_imgs = []
        filenames = []
        for i, image_path in enumerate(images_path):
            filename = os.path.basename(image_path)
            image = cvt_color(image_read(image_path))
            batch_imgs.append(image)
            filenames.append(filename)
            if (i + 1) % batch_size == 0 or i == len(images_path) - 1:
                # multi-scale inference (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
                results = infer_batch_scales(self.network, batch_imgs, scales, base_crop_size, flip)
                for ri, res in enumerate(results):
                    ori_img = batch_imgs[ri]
                    res_classes = np.unique(res).tolist()
                    t_classes = [r for r in res_classes if r in target_classes]
                    if t_classes:
                        for cls in t_classes:
                            if cls == 0:
                                continue
                            output_image = copy.deepcopy(ori_img)
                            output_image[res != cls] = [255, 255, 255]
                            output_image = Image.fromarray(output_image)
                            output_image.save(os.path.join(output_path, f"{NAME_CLASSES[cls]}_{filenames[ri]}"))
                        print(f"{filename} has been segmented to classes: {[NAME_CLASSES[r] for r in t_classes if r != 0]}.")
                    else:
                        print(f"{filename} has no target classes.")
                batch_imgs = []
                filenames = []


if __name__ == "__main__":
    model = DeepLabV3PlusMobileNetV2VOC2012(weight_path="./deeplab_mobilenetv2.ckpt", output_stride=16)
    model.infer("./data/", "./outputs/", scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75))
