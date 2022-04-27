"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=5990):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        layers = [conv_3x3_bn(1, 32, 2)]
        self.conv1 = nn.Sequential(*layers)
        # building inverted residual blocks
        # block = InvertedResidual
        # for t, c, n, s in self.cfgs:
        #     output_channel = _make_divisible(c , 8)
        #     for i in range(n):
        #         layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
        #         input_channel = output_channel
        self.f1 = nn.Sequential(InvertedResidual(32, 16, 1, 1))
        self.f2 = nn.Sequential(InvertedResidual(16, 24, 2, 6),
                                InvertedResidual(24, 24, 1, 6),
                                # nn.Dropout(0.2)
                                )
        self.f3 = nn.Sequential(InvertedResidual(24, 32, 2, 6),
                                InvertedResidual(32, 32, 1, 6),
                                InvertedResidual(32, 32, 1, 6),
                                # nn.Dropout(0.2)
                                )
        self.f4 = nn.Sequential(InvertedResidual(32, 64, 1, 6),
                                InvertedResidual(64, 64, 1, 6),
                                InvertedResidual(64, 64, 1, 6),
                                InvertedResidual(64, 64, 1, 6),
                                # nn.Dropout(0.2)
                                )
        self.f5 = nn.Sequential(InvertedResidual(64, 96, 1, 6),
                                InvertedResidual(96, 96, 1, 6),
                                InvertedResidual(96, 96, 1, 6),
                                # nn.Dropout(0.2)
                                )
        # self.f6 = nn.Sequential(InvertedResidual(96, 160, 1, 6),
        #                         InvertedResidual(160, 160, 1, 6),
        #                         InvertedResidual(160, 160, 1, 6))
        # self.f7 = nn.Sequential(InvertedResidual(160, 320, 1, 6))

        # layers.append(InvertedResidual(32, 16, 1, 1))
        
        # layers.append(InvertedResidual(16, 24, 2, 6))
        # layers.append(InvertedResidual(24, 24, 1, 6))

        # layers.append(InvertedResidual(24, 32, 2, 6))
        # layers.append(InvertedResidual(32, 32, 1, 6))
        # layers.append(InvertedResidual(32, 32, 1, 6))

        # layers.append(InvertedResidual(32, 64, 2, 6))
        # layers.append(InvertedResidual(64, 64, 1, 6))
        # layers.append(InvertedResidual(64, 64, 1, 6))
        # layers.append(InvertedResidual(64, 64, 1, 6))

        # layers.append(InvertedResidual(64, 96, 1, 6))
        # layers.append(InvertedResidual(96, 96, 1, 6))
        # layers.append(InvertedResidual(96, 96, 1, 6))

        # layers.append(InvertedResidual(96, 160, 2, 6))
        # layers.append(InvertedResidual(160, 160, 1, 6))
        # layers.append(InvertedResidual(160, 160, 1, 6))

        # layers.append(InvertedResidual(160, 320, 1, 6))

        # self.features = nn.Sequential(*layers)


        self.conv2 = conv_1x1_bn(96, 192)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(960, num_classes)#192*4

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        # print("1: ", x.shape)
        x = self.f1(x)
        # print("2: ", x.shape)
        x = self.f2(x)
        # print("3: ", x.shape)
        x = self.f3(x)
        # print("4: ", x.shape)
        x = self.f4(x)
        # print("5: ", x.shape)
        x = self.f5(x)
        # print("6: ", x.shape)
        # x = self.f6(x)
        # print("7: ", x.shape)
        # x = self.f7(x)
        # print("8: ", x.shape)
        x = self.conv2(x)
        # print("9: ", x.shape)
        # x = self.avgpool(x)
        x = x.permute(0, 3, 1, 2)
        # print("10: ", x.shape)
        # x = x.view(x.size(0), -1)
        x = x.view(x.shape[0], x.shape[1], -1)
        # print("11: ", x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == '__main__':

    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = MobileNetV2().cuda()
    print(summary(model, (1, 32, 280)))
