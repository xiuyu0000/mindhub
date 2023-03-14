import os
import multiprocessing
from typing import Optional, Callable

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.communication.management import init, get_rank

from mindhub.utils.images import read_dataset, image_read


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        init()
        rank_id = get_rank()
    else:
        rank_id = int(os.environ.get("RANK_ID", 0))

    return rank_size, rank_id


def create_dataset_imagenet_infer(dataset_path: str,
                                  transform: Optional[Callable] = None,
                                  batch_size: int = 1,
                                  num_parallel_workers: Optional[int] = None,
                                  ):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(str): the path of dataset.
        transform(optional(callable)): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32.
        num_parallel_workers(int): The number of workers signed to this task. Default: None.

    Returns:
        ds
    """

    device_num, rank_id = _get_rank_info()
    img_paths = read_dataset(dataset_path)
    img_list = [
        image_read(p)
        for p in img_paths
    ]

    if device_num == 1:
        data_set = ds.GeneratorDataset(img_list, column_names=["image"], num_parallel_workers=num_parallel_workers)
    else:
        data_set = ds.GeneratorDataset(img_list, column_names=["image"],num_parallel_workers=num_parallel_workers,
                                       num_shards=device_num, shard_id=rank_id)


    image_size = 224
    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if transform:
        transform_img = transform
    else:
        transform_img = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    cores = multiprocessing.cpu_count()
    if device_num == 1:
        num_parallel_workers = min(24, cores)
    else:
        num_parallel_workers = int(cores / device_num)

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=transform_img)
    # apply batch operations
    data_set = data_set.batch(batch_size)

    return data_set
