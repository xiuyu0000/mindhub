import multiprocessing
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision

from mindspore.communication.management import init, get_rank


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


def create_dataset_imagenet(dataset_path,
                            repeat_num=1,
                            transform=None,
                            num_parallel_workers=None,
                            shuffle=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle,
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

    transform_label = [C.TypeCast(mstype.int32)]

    cores = multiprocessing.cpu_count()
    if device_num == 1:
        num_parallel_workers = min(24, cores)
    else:
        num_parallel_workers = int(cores / device_num)

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=num_parallel_workers,
                            operations=transform_label)

    # apply batch operations
    data_set = data_set.batch(1)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
