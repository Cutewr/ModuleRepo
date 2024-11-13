'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        # 当前层输入通道数为64
        self.in_planes = 64

        # 卷积层 将输入3通道转换为输出64通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 归一化处理
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers 自底向上层
        # 输出通道数增加
        # 在提取特征的同时会进一步对特征图下采样，使特征图尺寸变小，语义增强，位置信息不明确
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        # 金字塔顶端层
        # 通过一个1*1的卷积将2048输入通道转换为256输出通道。降低通道数，以便后续与其他层进行融合处理
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        # 对经过处理后的特征图进行进一步的平滑处理，可能是为了减少一些由于采样或其他操作带来的锯齿状等不平整现象（抗 aliasing 效果）。
        # 通过这些平滑层的操作，可以使特征图更加平滑，有利于后续的特征融合和处理等操作。
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        # 关键！！
        # 实现不同层次特征图之间的连接和融合
        # 通过1*1卷积层将不同层次特征图的通道数都转换为256，便于后续的融合操作
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    # args：具体的层模块、目标通道数、块的数量、步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # 构建步长列表，第一个元素是stride，后面都是1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # 根据输入通道、输入通道、步长构建层
            layers.append(block(self.in_planes, planes, stride))
            # 每构建完层之后，使得输出通道乘上扩展系数expasion得到下一次的输入通道
            self.in_planes = planes * block.expansion
        # 将构造好的层组成一个Sequential并返回
        # *layers:解包layers中的参数layer，一个个传给Sequential
        return nn.Sequential(*layers)

    # 对两个特征图进行上采样相加
    # args：x【需要上采样的顶部特征图】 y【侧向特征图】
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        # _:是占位符，我们不关心y的前两个维度
        # 只关注高度和宽度
        _,_,H,W = y.size()
        # 使用双线性插值对x进行上采样，使得x尺寸调整为何y一样的(H,W) 再和y相加
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up 自底向上
        c1 = F.relu(self.bn1(self.conv1(x)))
        # 最大池化：提取数据的主要特征，减少计算量
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # 得到不同层次的特征图：越往上，语义信息越丰富，位置信息越不明确
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down 自顶向下
        # 将c5通过toplayer降低通道数为256，得到金字塔顶端特征图p5
        p5 = self.toplayer(c5)
        # latlayer：将中间层的通道数转换为256
        # _upsample_add:调整顶端大小，和中间层相加
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

test()