import torch
import torch.nn as nn
from fontTools.unicodedata import block


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)   # (3,64,32,32)
        attn2 = self.conv_spatial(attn1)    # (3,64,32,32)

        attn1 = self.conv1(attn1)   # (3,32,32,32)
        attn2 = self.conv2(attn2)   # (3,32,32,32)

        attn = torch.cat([attn1, attn2], dim=1)     # (3,64,32,32)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)    # (3,1,32,32) 对通道维度做均值
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)     # (3,1,32,32) 对通道维度取最大值
        agg = torch.cat([avg_attn, max_attn], dim=1)    # (3,2,32,32)
        sig = self.conv_squeeze(agg).sigmoid()   # (3,2,32,32) 再使用sigmoid函数将实数映射到(0,1)上
        # 整行代码的作用是根据从 sig 中获取的两个不同通道的注意力权重，分别对 attn1 和 attn2 进行加权操作（乘法运算），
        # 然后将这两个加权后的结果相加，得到一个融合了两种不同特征且考虑了注意力权重的新特征张量 attn。
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)  # (3,32,32,32)
        attn = self.conv(attn)    # (3,64,32,32)
        return x * attn

if __name__ == '__main__':
    block = LSKblock(64)    # 输入维度
    input = torch.rand(3,64,32,32)  #输入B C H W
    output = block(input)
    print(output.shape)    # (3,64,32,32)