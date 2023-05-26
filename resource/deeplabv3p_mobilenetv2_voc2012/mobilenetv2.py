from mindspore import nn

BatchNorm2d = nn.BatchNorm2d


def conv_bn(inp, oup, stride):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, 3, stride, pad_mode="pad", padding=1),
        BatchNorm2d(oup),
        nn.ReLU6()
    ])


def conv_1x1_bn(inp, oup):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, 1, 1, pad_mode="pad", padding=0),
        BatchNorm2d(oup),
        nn.ReLU6()
    ])


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.SequentialCell([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad_mode="pad", padding=1, group=hidden_dim),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                nn.Conv2d(hidden_dim, oup, 1, 1, pad_mode="pad", padding=0),
                BatchNorm2d(oup),
            ])
        else:
            self.conv = nn.SequentialCell([
                nn.Conv2d(inp, hidden_dim, 1, 1, pad_mode="pad", padding=0),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad_mode="pad", padding=1, group=hidden_dim),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                nn.Conv2d(hidden_dim, oup, 1, 1, pad_mode="pad", padding=0),
                BatchNorm2d(oup)
            ])

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Cell):
    def __init__(self, n_class=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2],  # 32, 32, 96 -> 16, 16, 160     14
            [6, 320, 1, 1],  # 16, 16, 160 -> 16, 16, 320
        ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for ni in range(n):
                if ni == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.SequentialCell(self.features)

        self.classifier = nn.SequentialCell([
            nn.Dropout(p=0.2),
            nn.Dense(self.last_channel, n_class),
        ])

    def construct(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MobileNetV2()
    for i, layer in enumerate(model.features):
        print(i, layer)
