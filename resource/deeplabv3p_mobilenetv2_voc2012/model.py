from mindspore import nn, ops

from mobilenetv2 import MobileNetV2


def _no_stride_dilate(m, dilate):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate // 2, dilate // 2)
                m.padding = (dilate // 2, dilate // 2)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)


class MobileNetV2BackBone(nn.Cell):
    def __init__(self, downsample_factor=8):
        super(MobileNetV2BackBone, self).__init__()
        from functools import partial

        model = MobileNetV2()
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(_no_stride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(_no_stride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(_no_stride_dilate, dilate=2)
                )

    def construct(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Cell):
    def __init__(self, dim_in, dim_out, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.SequentialCell([
            nn.Conv2d(dim_in, dim_out, 1, 1, pad_mode="pad", padding=0, dilation=rate, has_bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        ])
        self.branch2 = nn.SequentialCell([
            nn.Conv2d(dim_in, dim_out, 3, 1, pad_mode="pad", padding=6 * rate, dilation=6 * rate, has_bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        ])
        self.branch3 = nn.SequentialCell([
            nn.Conv2d(dim_in, dim_out, 3, 1, pad_mode="pad", padding=12 * rate, dilation=12 * rate, has_bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        ])
        self.branch4 = nn.SequentialCell([
            nn.Conv2d(dim_in, dim_out, 3, 1, pad_mode="pad", padding=18 * rate, dilation=18 * rate, has_bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        ])
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, pad_mode="pad", padding=0, has_bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU()

        self.conv_cat = nn.SequentialCell([
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, pad_mode="pad", padding=0, has_bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        ])

    def construct(self, x):
        _, _, row, col = x.shape

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = ops.mean(x, 2, True)
        global_feature = ops.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = ops.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = ops.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], axis=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Cell):
    def __init__(self, num_classes, downsample_factor=16):
        super(DeepLab, self).__init__()

        self.backbone = MobileNetV2BackBone(downsample_factor=downsample_factor)
        in_channels = 320
        low_level_channels = 24

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.SequentialCell([
            nn.Conv2d(low_level_channels, 48, 1, pad_mode="pad", padding=0, has_bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU()
        ])

        self.cat_conv = nn.SequentialCell([
            nn.Conv2d(48 + 256, 256, 3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 256, 3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Dropout(p=0.1),
        ])
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, x):
        h, w = x.shape[2:]
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = ops.interpolate(x, size=(low_level_features.shape[2], low_level_features.shape[3]), mode='bilinear',
                            align_corners=True)
        x = self.cat_conv(ops.cat((x, low_level_features), axis=1))
        x = self.cls_conv(x)
        x = ops.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    import mindspore as ms
    import torch
    import pandas as pd
    from collections import defaultdict
    from copy import deepcopy

    net = DeepLab(21)
    pt_param = torch.load("./deeplab_mobilenetv2.pth", map_location=torch.device('cpu'))
    pt_param_names = list(pt_param.keys())
    for pn in pt_param_names:
        if "num_batches_tracked" in pn:
            pt_param.pop(pn)

    ms_param = net.parameters_dict()
    # df_dict = defaultdict(list)
    # for name, param in ms_param.items():
    #     df_dict["ms_name"].append(name)
    #     df_dict["ms_shape"].append(param.shape)
    #
    # for name, param in pt_param.items():
    #     df_dict["pt_name"].append(name)
    #     df_dict["pt_shape"].append(param.shape)
    #
    # df = pd.DataFrame(df_dict)
    # df.to_csv("./compare_param.csv", index=False)
    ms_param_names = list(ms_param.keys())
    ms2pt = {"moving_mean": "running_mean", "moving_variance": "running_var", "gamma": "weight", "beta": "bias"}
    for mn in ms_param_names:
        ori_name = deepcopy(mn)
        param = pt_param.get(mn)
        if param is None:
            post_name = mn.split(".")[-1]
            if post_name in ms2pt:
                mn = mn.replace(post_name, ms2pt[post_name])
            param = pt_param[mn]
        ms_param[ori_name] = ms.Parameter(ms.Tensor(param.detach().numpy()), name=ori_name)
    ms.load_param_into_net(net, ms_param)
    ms.save_checkpoint(net, "./deeplab_mobilenetv2.ckpt")


