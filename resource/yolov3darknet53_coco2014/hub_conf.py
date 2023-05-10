import os
from typing import Optional
from download import download
import mindspore as ms

from mindhub.models.registry import register_model
from mindhub.utils.images import image_read

from model import YOLOV3DarkNet53
from detection import DetectionEngine
from preprocess import create_yolo_infer_dataset


def load_parameters(network, file_name):
    print(f"yolov3 pretrained network model: {file_name}")
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    print(ms.load_param_into_net(network, param_dict_new))
    print(f'load_model {file_name} success')


@register_model(model_name="yolov3darknet53_coco2014",
                model_type="image/detection",
                paper="https://pjreddie.com/media/files/papers/YOLOv3.pdf",
                pretrained=True)
class YoloV3DarkNet53COCO2014:
    """YoloV3DarkNet53 trained by using COCO2014."""
    def __init__(self,
                 model_name: str = "yolov3darknet53_coco2014",
                 pretrained: bool = False,
                 weight_path: Optional[str] = None):

        self.network = YOLOV3DarkNet53(is_training=False) \
            if model_name == "yolov3darknet53_coco2014" else None
        self.ckpt_url = "https://download.mindspore.cn/models/r1.9/yolov3darknet53shape416_ascend_" \
                        "v190_coco2014_official_cv_map31.8.ckpt"
        self.ckpt_path = "./yolov3darknet53shape416_ascend_v190_coco2014_official_cv_map31.8.ckpt"

        if weight_path:
            load_parameters(self.network, weight_path)
        elif pretrained:
            path = download(self.ckpt_url, self.ckpt_path, replace=False)
            load_parameters(self.network, path)

    def infer(self, data_dir, outputs_dir, batch_size=1):
        os.makedirs(outputs_dir, exist_ok=True)
        self.network.set_train(False)
        infer_dataset, file_names = create_yolo_infer_dataset(data_dir, batch_size=batch_size)
        res = []
        for i, (img, image_shape) in enumerate(infer_dataset.create_tuple_iterator()):
            output_l, output_m, output_s = self.network(img)
            output_l = output_l.asnumpy()
            output_m = output_m.asnumpy()
            output_s = output_s.asnumpy()
            image_shape = image_shape.asnumpy()
            for bi in range(batch_size):
                save_path = os.path.join(outputs_dir, file_names[i * batch_size + bi])
                detection = DetectionEngine()
                detection.detect([output_l[bi], output_m[bi], output_s[bi]], image_shape[bi])
                detection.do_nms_for_results()
                img_ori = image_read(os.path.join(data_dir, file_names[i * batch_size + bi]))
                detection.save_bbox_img(img_ori, save_path)
                print(f'No.{i * batch_size + bi + 1} image inference result has been saved in {save_path}\n'
                      f'Predict Result: {detection.det_boxes}.')
                res.append({"det_boxes": detection.det_boxes, "output_save_path": save_path})
        return res


if __name__ == '__main__':
    data_path = "./data/"
    outputs_dir = "./outputs/"
    batch_size = 1
    os.makedirs(outputs_dir, exist_ok=True)
    model = YoloV3DarkNet53COCO2014(pretrained=True)
    outputs_info = model.infer(data_path, outputs_dir, batch_size=2)
