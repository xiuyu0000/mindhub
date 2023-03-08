# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""post process for 310 inference"""
import os
from scipy import io


def index2label(data_path):
    """Dictionary output for image numbers and categories of the ImageNet dataset."""
    metafile = os.path.join(data_path, "ILSVRC2012_devkit_t12/data/meta.mat")
    meta = io.loadmat(metafile, squeeze_me=True)['synsets']

    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

    _, wnids, classes = list(zip(*meta))[:3]
    clssname = [tuple(clss.split(', ')) for clss in classes]
    wnid2class = {wnid: clss for wnid, clss in zip(wnids, clssname)}
    wind2class_name = sorted(wnid2class.items(), key=lambda x: x[0])

    mapping = {}
    for index, (_, class_name) in enumerate(wind2class_name):
        mapping[index] = class_name[0]
    return mapping