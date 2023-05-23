"""fcn8s_vgg16 model"""
import mindspore.nn as nn
import mindspore.ops as ops


class FCN8s(nn.Cell):
    def __init__(self, n_class):
        """
        FCN8S network
        n_class: the number of the class
        """
        super().__init__()
        self.add_strategy = None
        self.maxpool_strategy = None
        self.relu_strategy = None
        self.bn_strategy = None
        self.conv2d_strategy = None
        self.n_class = n_class
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=4096,
                      kernel_size=7, weight_init='xavier_uniform'),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
        )
        self.conv7 = nn.SequentialCell(
            nn.Conv2d(in_channels=4096, out_channels=4096,
                      kernel_size=1, weight_init='xavier_uniform'),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
        )
        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=self.n_class,
                                  kernel_size=1, weight_init='xavier_uniform')
        self.upscore2 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                           kernel_size=4, stride=2, weight_init='xavier_uniform')
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=self.n_class,
                                     kernel_size=1, weight_init='xavier_uniform')
        self.upscore_pool4 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                                kernel_size=4, stride=2, weight_init='xavier_uniform')
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=self.n_class,
                                     kernel_size=1, weight_init='xavier_uniform')
        self.upscore8 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                           kernel_size=16, stride=8, weight_init='xavier_uniform')
        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self.add1 = ops.Add()
        self.add2 = ops.Add()

    def set_model_parallel_shard_strategy(self, device_num):
        """
        set model parallel
        device_num: the number of the device
        """

        self.conv2d_strategy = ((1, 1, 1, device_num), (1, 1, 1, 1))
        self.bn_strategy = ((1, 1, 1, device_num), (1,), (1,), (1,), (1,))
        self.relu_strategy = ((1, 1, 1, device_num),)
        self.maxpool_strategy = ((1, 1, 1, device_num),)
        self.add_strategy = ((1, 1, 1, device_num), (1, 1, 1, device_num))

        self.conv1.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv1.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv1.cell_list[2].relu.shard(self.relu_strategy)
        self.conv1.cell_list[3].conv2d.shard(self.conv2d_strategy)
        self.conv1.cell_list[4].bn_train.shard(self.bn_strategy)
        self.conv1.cell_list[5].relu.shard(self.relu_strategy)
        self.pool1.max_pool.shard(self.maxpool_strategy)
        self.conv2.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv2.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv2.cell_list[2].relu.shard(self.relu_strategy)
        self.conv2.cell_list[3].conv2d.shard(self.conv2d_strategy)
        self.conv2.cell_list[4].bn_train.shard(self.bn_strategy)
        self.conv2.cell_list[5].relu.shard(self.relu_strategy)
        self.pool2.max_pool.shard(self.maxpool_strategy)
        self.conv3.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv3.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv3.cell_list[2].relu.shard(self.relu_strategy)
        self.conv3.cell_list[3].conv2d.shard(self.conv2d_strategy)
        self.conv3.cell_list[4].bn_train.shard(self.bn_strategy)
        self.conv3.cell_list[5].relu.shard(self.relu_strategy)
        self.conv3.cell_list[6].conv2d.shard(self.conv2d_strategy)
        self.conv3.cell_list[7].bn_train.shard(self.bn_strategy)
        self.conv3.cell_list[8].relu.shard(self.relu_strategy)
        self.pool3.max_pool.shard(self.maxpool_strategy)
        self.conv4.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv4.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv4.cell_list[2].relu.shard(self.relu_strategy)
        self.conv4.cell_list[3].conv2d.shard(self.conv2d_strategy)
        self.conv4.cell_list[4].bn_train.shard(self.bn_strategy)
        self.conv4.cell_list[5].relu.shard(self.relu_strategy)
        self.conv4.cell_list[6].conv2d.shard(self.conv2d_strategy)
        self.conv4.cell_list[7].bn_train.shard(self.bn_strategy)
        self.conv4.cell_list[8].relu.shard(self.relu_strategy)
        self.pool4.max_pool.shard(self.maxpool_strategy)
        self.conv5.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv5.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv5.cell_list[2].relu.shard(self.relu_strategy)
        self.conv5.cell_list[3].conv2d.shard(self.conv2d_strategy)
        self.conv5.cell_list[4].bn_train.shard(self.bn_strategy)
        self.conv5.cell_list[5].relu.shard(self.relu_strategy)
        self.conv5.cell_list[6].conv2d.shard(self.conv2d_strategy)
        self.conv5.cell_list[7].bn_train.shard(self.bn_strategy)
        self.conv5.cell_list[8].relu.shard(self.relu_strategy)
        self.pool5.max_pool.shard(((1, 1, 1, device_num),))
        self.conv6.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv6.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv6.cell_list[2].relu.shard(self.relu_strategy)
        self.conv7.cell_list[0].conv2d.shard(self.conv2d_strategy)
        self.conv7.cell_list[1].bn_train.shard(self.bn_strategy)
        self.conv7.cell_list[2].relu.shard(self.relu_strategy)
        self.score_fr.conv2d.shard(self.conv2d_strategy)
        self.upscore2.conv2d_transpose.shard(self.conv2d_strategy)
        self.score_pool4.conv2d.shard(self.conv2d_strategy)
        self.upscore_pool4.conv2d_transpose.shard(self.conv2d_strategy)
        self.score_pool3.conv2d.shard(self.conv2d_strategy)
        self.upscore8.conv2d_transpose.shard(self.conv2d_strategy)
        self.add1.shard(self.add_strategy)
        self.add2.shard(self.add_strategy)

    def construct(self, x):
        """
        construct method
        """
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        x5 = self.conv5(p4)
        p5 = self.pool5(x5)

        x6 = self.conv6(p5)
        x7 = self.conv7(x6)

        sf = self.score_fr(x7)
        u2 = self.upscore2(sf)

        s4 = self.score_pool4(p4)
        f4 = self.add1(s4, u2)
        u4 = self.upscore_pool4(f4)

        s3 = self.score_pool3(p3)
        f3 = self.add2(s3, u4)
        out = self.upscore8(f3)

        return out
