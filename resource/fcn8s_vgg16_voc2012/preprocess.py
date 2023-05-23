import cv2
import numpy as np

import mindspore as ms
from mindspore import Tensor

#################### 1. Change the following line ####################
from mindhub.utils import image_read
from PIL import Image
import mindspore.ops as ops
import copy
#################### 1. Change the above line ####################

IMG_MEAN = [103.53, 116.28, 123.675]
IMG_STD = [57.375, 57.120, 58.395]


def cvt_color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(img_, crop_size=513):
    """pre_process"""
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(IMG_MEAN)
    image_std = np.array(IMG_STD)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def infer_batch(infer_net, img_lst, crop_size=513, flip=True):
    """infer batch"""
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for i in range(batch_size):
        img_ = img_lst[i]
        img_, resize_h, resize_w = pre_process(img_, crop_size)
        batch_img[i] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = infer_net(Tensor(batch_img, ms.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = infer_net(Tensor(batch_img, ms.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def infer_batch_scales(infer_net, img_lst, scales, base_crop_size=513, flip=True):
    """eval_batch_scales"""
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = infer_batch(infer_net, img_lst, crop_size=sizes_[0], flip=flip)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = infer_batch(infer_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=-1))
    return result_msk


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize(size=(nw, nh), resample=Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def preprocess_input(image):
    image /= 255.0
    return image


def infer_image(infer_net, img_path):
    """eval_image"""
    colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]
    img = Image.open(img_path)
    ori_img = copy.deepcopy(img)
    img = cvt_color(img)
    ori_h, ori_w, _ = np.array(img).shape
    image_data, nw, nh = resize_image(img, (512, 512))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    images = ms.Tensor(image_data, ms.float32)
    pr = infer_net(images)[0]
    pr = ops.softmax(pr.permute(1, 2, 0), axis=-1).asnumpy()
    pr = pr[int((512 - nh) // 2): int((512 - nh) // 2 + nh),
            int((512 - nw) // 2): int((512 - nw) // 2 + nw)]
    pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    pr = pr.argmax(axis=-1)
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [ori_h, ori_w, -1])
    image = Image.fromarray(np.uint8(seg_img))
    image = Image.blend(ori_img, image, 0.7)
    image.save("./result.jpg")
