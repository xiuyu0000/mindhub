""" DarkNet53 network for YOLOv3."""

import easydict

import mindspore as ms
from mindspore import ops, nn

DEFAULT_CONFIG = easydict.EasyDict({
    "keep_detect": True,
    "backbone_layers": [1, 2, 8, 8, 4],
    "backbone_input_shape": [32, 64, 128, 256, 512],
    "backbone_shape": [64, 128, 256, 512, 1024],
    "out_channel": 255,
    "anchor_scales": [[10, 13],
                      [16, 30],
                      [33, 23],
                      [30, 61],
                      [62, 45],
                      [59, 119],
                      [116, 90],
                      [156, 198],
                      [373, 326]],
    "num_classes": 80,
    "eval_ignore_threshold": 0.001,
    "nms_thresh": 0.5,
})


def _make_layer(block, layer_num, in_channel, out_channel):
    """
    Make Layer for DarkNet.
    :param block: Cell. DarkNet block.
    :param layer_num: Integer. Layer number.
    :param in_channel: Integer. Input channel.
    :param out_channel: Integer. Output channel.
    Examples:
        _make_layer(ConvBlock, 1, 128, 256)
    """
    layers = []
    darkblk = block(in_channel, out_channel)
    layers.append(darkblk)

    for _ in range(1, layer_num):
        darkblk = block(out_channel, out_channel)
        layers.append(darkblk)

    return nn.SequentialCell(layers)


def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.9,
                  eps=1e-5,
                  pad_mode="same"):
    """Get a conv2d, batchnorm and relu layer"""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum, eps=eps),
         nn.LeakyReLU(alpha)]
    )


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    """Get a conv2d, batchnorm and relu layer"""
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.1),
         nn.ReLU()]
    )


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.
    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(3, 208)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualBlock, self).__init__()
        out_chls = out_channels // 2
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = ops.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


class DarkNet(nn.Cell):
    """
    DarkNet V1 network.
    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        detect: Bool. Whether detect or not. Default:False.
    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3,f4,f5).
    Examples:
        DarkNet(ResidualBlock,
               [1, 2, 8, 8, 4],
               [32, 64, 128, 256, 512],
               [64, 128, 256, 512, 1024],
               100)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 detect=False):
        super(DarkNet, self).__init__()

        self.outchannel = out_channels[-1]
        self.detect = detect

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 5:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 5!")
        self.conv0 = conv_block(3,
                                in_channels[0],
                                kernel_size=3,
                                stride=1)
        self.conv1 = conv_block(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=2)
        self.layer1 = _make_layer(block,
                                  layer_nums[0],
                                  in_channel=out_channels[0],
                                  out_channel=out_channels[0])
        self.conv2 = conv_block(in_channels[1],
                                out_channels[1],
                                kernel_size=3,
                                stride=2)
        self.layer2 = _make_layer(block,
                                  layer_nums[1],
                                  in_channel=out_channels[1],
                                  out_channel=out_channels[1])
        self.conv3 = conv_block(in_channels[2],
                                out_channels[2],
                                kernel_size=3,
                                stride=2)
        self.layer3 = _make_layer(block,
                                  layer_nums[2],
                                  in_channel=out_channels[2],
                                  out_channel=out_channels[2])
        self.conv4 = conv_block(in_channels[3],
                                out_channels[3],
                                kernel_size=3,
                                stride=2)
        self.layer4 = _make_layer(block,
                                  layer_nums[3],
                                  in_channel=out_channels[3],
                                  out_channel=out_channels[3])
        self.conv5 = conv_block(in_channels[4],
                                out_channels[4],
                                kernel_size=3,
                                stride=2)
        self.layer5 = _make_layer(block,
                                  layer_nums[4],
                                  in_channel=out_channels[4],
                                  out_channel=out_channels[4])

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)
        if self.detect:
            return c7, c9, c11

        return c11

    def get_out_channels(self):
        return self.outchannel


def darknet53():
    """
    Get DarkNet53 neural network.
    Returns:
        Cell, cell instance of DarkNet53 neural network.
    Examples:
        darknet53()
    """
    return DarkNet(ResidualBlock, [1, 2, 8, 8, 4],
                   [32, 64, 128, 256, 512],
                   [64, 128, 256, 512, 1024])


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.
    Args:
        in_channels: Integer. Input channel.
        out_chls: Integer. Middle channel.
        out_channels: Integer. Output channel.
    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).
    Examples:
        YoloBlock(1024, 512, 255)
    """

    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class YOLOv3(nn.Cell):
    """
     YOLOv3 Network.
     Note:
         backbone = darknet53
     Args:
         backbone_shape: List. Darknet output channels shape.
         backbone: Cell. Backbone Network.
         out_channel: Integer. Output channel.
     Returns:
         Tensor, output tensor.
     Examples:
         YOLOv3(backbone_shape=[64, 128, 256, 512, 1024]
                backbone=darknet53(),
                out_channel=255)
     """

    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2] // 2, ksize=1)
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2] + backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3] // 2, ksize=1)
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3] + backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        img_hight = ops.Shape()(x)[2]
        img_width = ops.Shape()(x)[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = ops.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)
        ups2 = ops.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class DetectionBlock(nn.Cell):
    """
     YOLOv3 detection Network. It will finally output the detection result.
     Args:
         scale: Character.
         config: Configuration.
         is_training: Bool, Whether train or not, default True.
     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).
     Examples:
         DetectionBlock(scale='l',stride=32,config=config)
     """

    def __init__(self, scale, config=None, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.conf_training = is_training

    def construct(self, x, input_shape):
        num_batch = ops.Shape()(x)[0]
        grid_size = ops.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = ops.Reshape()(x, (num_batch, self.num_anchors_per_scale, self.num_attrib,
                                       grid_size[0], grid_size[1]))
        prediction = ops.Transpose()(prediction, (0, 3, 4, 1, 2))

        range_x = tuple(range(grid_size[1]))
        range_y = tuple(range(grid_size[0]))
        grid_x = ops.Cast()(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.Cast()(ops.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.sigmoid(box_xy) + grid) / ops.Cast()(ops.tuple_to_array((grid_size[1],
                                                                                grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = ops.Exp()(box_wh) * self.anchors / input_shape

        if self.conf_training:
            return grid, prediction, box_xy, box_wh
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)
        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class YOLOV3DarkNet53(nn.Cell):
    """
    Darknet based YOLOV3 network.
    Args:
        is_training: Bool. Whether train or not.
        config: Optional. The config settings.
    Returns:
        Cell, cell instance of Darknet based YOLOV3 neural network.
    Examples:
        YOLOV3DarkNet53(True)
    """

    def __init__(self, is_training, config=None):
        super(YOLOV3DarkNet53, self).__init__()
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config
        self.keep_detect = self.config.keep_detect
        self.tenser_to_array = ops.TupleToArray()

        # YOLOv3 network
        self.feature_map = YOLOv3(backbone=DarkNet(ResidualBlock, self.config.backbone_layers,
                                                   self.config.backbone_input_shape,
                                                   self.config.backbone_shape,
                                                   detect=True),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training, config=self.config)
        self.detect_2 = DetectionBlock('m', is_training=is_training, config=self.config)
        self.detect_3 = DetectionBlock('s', is_training=is_training, config=self.config)

    def construct(self, x):
        input_shape = ops.shape(x)[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape), ms.float32)
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            return big_object_output, medium_object_output, small_object_output
        output_big = self.detect_1(big_object_output, input_shape)
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)

        return output_big, output_me, output_small
