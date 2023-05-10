"""Yolo dataset distributed sampler."""
import os.path
import random

import numpy as np
from PIL import Image
import cv2
import mindspore.dataset as ds

from mindhub.utils.images import read_dataset, image_read

MIN_KEYPOINTS_PRE_IMAGE = 10
INFER_SHAPE = [416, 416]


def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img / 255.
    # Computed from random subset of ImageNet training images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img


def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and an auto-estimation method.
    Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic or Bilinear.
    Args:
        interp (int): Interpolation method for all resizing operations.
            - 0: Nearest Neighbors Interpolation.
            - 1: Bilinear interpolation.
            - 2: Bicubic interpolation over 4x4 pixel neighborhood.
            - 3: Nearest Neighbors. Originally it should be Area-based, as we cannot find Area-based,
              so we use NN instead. Area-based (resampling using pixel area relation).
              It may be a preferred method for image decimation, as it gives moire-free results.
              But when the image is zoomed, it is similar to the Nearest Neighbors method. (used by default).
            - 4: Lanczos interpolation over 8x8 pixel neighborhood.
            - 9: Cubic for enlarge, area for shrink, bilinear for others.
            - 10: Random select from interpolation method mentioned above.
        sizes (tuple): Format should like (old_height, old_width, new_height, new_width),
            if None provided, auto(9) will return Area(2) anyway. Default: ()
    Returns:
        int, interp method from 0 to 4.
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def pil_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]


def _reshape_data(image, image_size):
    """Reshape image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    # original image shape fir:H sec:W
    h, w = image_size
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    image = image.resize((w, h), pil_image_reshape(interp))
    image_data = statistic_normalize_img(image, statistic_norm=True)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
    image_data = image_data.astype(np.float32)
    return image_data, ori_image_shape


def reshape_fn(image):
    input_size = INFER_SHAPE
    image, ori_image_shape = _reshape_data(image, image_size=input_size)
    return image, ori_image_shape


class COCOYoloInferDataset:
    """YOLOV3 Infer Dataset"""

    def __init__(self, img_path: str):
        self._path = read_dataset(img_path)
        self.file_name = [os.path.basename(path) for path in self._path]

    def __getitem__(self, index):
        item = image_read(self._path[index])
        return item

    def __len__(self):
        return len(self._path)


def create_yolo_infer_dataset(image_dir, batch_size=1, shuffle=False):
    """Create dataset for YOLOV3."""
    cv2.setNumThreads(0)

    yolo_dataset = COCOYoloInferDataset(img_path=image_dir)
    hwc_to_chw = ds.vision.HWC2CHW()
    dataset = ds.GeneratorDataset(yolo_dataset, column_names=["image"], shuffle=shuffle)

    dataset = dataset.map(operations=reshape_fn, input_columns=["image"],
                          output_columns=["image", "image_shape"],
                          num_parallel_workers=8)
    dataset = dataset.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=8)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, yolo_dataset.file_name
