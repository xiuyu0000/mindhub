"""image process."""
import os
from typing import List
import pathlib

import numpy as np
from PIL import Image

from mindhub.utils.path import check_file_exist

__all__ = [
    "read_dataset",
    "image_read",
]

_IMAGE_FORMAT = (
    '.JPEG',
    '.jpeg',
    '.PNG',
    '.png',
    '.JPG',
    '.jpg',
    '.PPM',
    '.ppm',
    '.BMP',
    '.bmp',
    '.PGM',
    '.pgm',
    '.WEBP',
    '.webp',
    '.TIF',
    '.tif',
    '.TIFF',
    '.tiff'
)


def read_dataset(path: str) -> List[str]:
    """
    Get the path list and index list of images.
    """
    img_list = list()

    if os.path.isdir(path):
        for img_name in os.listdir(path):
            if pathlib.Path(img_name).suffix in _IMAGE_FORMAT:
                img_path = os.path.join(path, img_name)
                img_list.append(img_path)
    else:
        img_list.append(path)

    return img_list


def image_read(image: str) -> np.array:
    """
    Read an image.

    Args:
        image (ndarray or str or Path): Ndarray, str or pathlib.Path.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        check_file_exist(image)
        image = Image.open(image)
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return np.asarray(image)
