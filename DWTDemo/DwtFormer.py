import math

import torch
from timm.models import trunc_normal_
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn.functional as F

from DWTDemo.Dcm import DCM
from DWTDemo.Dwt import DWTForward, SWTInverse



class RLN(nn.Module):
	r"""Revised LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias


class DustFormer(nn.Module):
	def __init__(self, in_chans=3, out_chans=3, window_size=8,
				 embed_dims=[12, 48, 192, 48, 12],
				 mlp_ratios=[2., 4., 4., 2., 2.],
				 depths=[16, 16, 16, 8, 8],
				 num_heads=[2, 4, 6, 1, 1],
				 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
				 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
				 norm_layer=[RLN, RLN, RLN, RLN, RLN]):

		super(DustFormer, self).__init__()

		# setting
		self.patch_size = 4
		self.window_size = window_size
		self.mlp_ratios = mlp_ratios

		# split image into non-overlapping patches
		self.patch_embed = DWT ()
			#patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layer1 = DwtViTBlock(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
								 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
								 norm_layer=norm_layer[0], window_size=window_size,
								 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

		self.patch_merge1 = DWT()
			#patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

		self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

		self.layer2 = DwtViTBlock(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
								 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
								 norm_layer=norm_layer[1], window_size=window_size,
								 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

		self.patch_merge2 = DWT()
			#patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

		self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

		self.layer3 = DwtViTBlock(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
								 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
								 norm_layer=norm_layer[2], window_size=window_size,
								 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

		self.patch_split1 = IDWT()
			#patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

		#assert embed_dims[1] == embed_dims[3]
		self.fusion1 =CIFM(embed_dims[3])


		self.layer4 = DwtViTBlock(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
								 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
								 norm_layer=norm_layer[3], window_size=window_size,
								 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

		self.patch_split2 = IDWT()
			#patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

		#assert embed_dims[0] == embed_dims[4]
		self.fusion2 = CIFM(embed_dims[4])

		self.layer5 = DwtViTBlock(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
								 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
								 norm_layer=norm_layer[4], window_size=window_size,
								 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

		# merge non-overlapping patches into image
		self.patch_unembed = IDWT()
			#patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
		self.dcm = DCM(192, [3, 6, 9])

	def check_image_size(self, x):
		# NOTE: for I2I test
		_, _, h, w = x.size()
		mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
		mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
		x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward_features(self, x):
		x = self.patch_embed(x)
		x = self.layer1(x)
		skip1 = x

		x = self.patch_merge1(x)
		x = self.layer2(x)
		skip2 = x

		x = self.patch_merge2(x)
		#x = self.layer3(x)
		x = self.dcm(x)
		#print(x.shape)
		x = self.patch_split1(x)

		x = self.fusion1([x, self.skip2(skip2)]) + x
		x = self.layer4(x)
		x = self.patch_split2(x)

		x = self.fusion2([x, self.skip1(skip1)]) + x
		x = self.layer5(x)
		x = self.patch_unembed(x)
		return x


	def forward(self, x):
		#H, W = x.shape[2:]
		x = self.check_image_size(x)

		x = self.forward_features(x)#
		return x

# 包装
class DWT(nn.Module):
	def __init__(self):
		super().__init__()
		self.dwt = DWTForward(J=1, wave='db1', mode='zero')

	def forward(self, x):
		# 得到低频部分yl和高频系数列表yh。其中yh是一个长度为J的列表，第一个元素是最精细尺度的高频系数。
		yl, yh = self.dwt(x)
		#print(yl.size())
		# 获取低频部分yl的维度信息，包括批次大小b、通道数c、高度h和宽度w。
		b, c, h, w = yl.size()
		# 对于高频系数列表yh中的每个元素：
		# 首先使用permute方法对维度进行调整，然后使用reshape方法将其形状调整为(b, c * 3, h, w)。
		# 这里的c * 3表示将三个方向（水平、垂直、对角）的高频系数合并为一个张量，每个方向对应c个通道。
		for i in range(len(yh)):
			yh_re = yh[i].permute(0, 2, 1, 3, 4).reshape(b, c * 3, h, w)
		# 将调整后的低频部分yl和高频系数（经过处理后的yh_re）在通道维度上进行拼接，得到最终的输出张量y并返回。
		y = torch.cat((yl, yh_re), 1)
		#print(y.size())
		return y


# 利用离散小波变换将图像分解为不同频率的子带
class DwtViTBlock(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, window_size=8,
				 attn_ratio=0., attn_loc='last', conv_type=None):

		super().__init__()
		self.dim = dim
		self.depth = depth

		attn_depth = attn_ratio * depth

		if attn_loc == 'last':
			use_attns = [i >= depth-attn_depth for i in range(depth)]
		elif attn_loc == 'first':
			use_attns = [i < attn_depth for i in range(depth)]
		elif attn_loc == 'middle':
			use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

		# build blocks
		self.blocks = nn.ModuleList([
			TransformerBlock(network_depth=network_depth,
							 dim=dim,
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 window_size=window_size,
							 shift_size=0 if (i % 2 == 0) else window_size // 2,
							 use_attn=use_attns[i], conv_type=conv_type)
			for i in range(depth)])

	def forward(self, x):
		# 依次经过TransformerBlock
		for blk in self.blocks:
			x = blk(x)
		return x

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
		# 残差连接准备
        identity = x
		# 当启用注意力机制时，先归一化
        if self.use_attn: x, rescale, rebias = self.norm1(x)
		# 再传入Attention层
        x = self.attn(x)
		# 首先将经过注意力计算后的 x 乘以 rescale 并加上 rebias
		# 为了将经过归一化和注意力计算后的结果调整回与原始数据在分布上更相近的状态。因为前面的归一化操作可能改变了数据的原始分布，通过这一步可以在一定程度上恢复。
        if self.use_attn: x = x * rescale + rebias
		# 将调整后的 x 与之前保存的原始输入数据 identity 进行相加，实现残差连接。
		# 这样即使注意力机制的计算结果不理想或者出现错误，网络仍然能够保留一部分原始输入的信息，使得网络的输出不至于完全偏离原始输入，同时也有助于网络学习到更有效的特征表示。
        x = identity + x

		# 多层感知机处理
        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x

class Attention(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = network_depth
		self.use_attn = use_attn
		self.conv_type = conv_type

		if self.conv_type == 'Conv':
			self.conv = nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
				nn.ReLU(True),
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
			)

        # groups=dim的设置意味着这是一个深度可分离卷积（Depthwise Separable Convolution）操作。
        # 在深度可分离卷积中，输入特征图的每个通道会单独进行卷积运算(与普通卷积相比，普通卷积是对所有通道同时进行卷积)
        # 这样可以在一定程度上减少计算量，同时又能提取到不同通道的独特特征，对于某些需要关注通道级别的特征提取任务可能更为合适。
        # 填充模式同样为'reflect'，用于处理边界情况。
		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			# 1*1卷积调整通道维度
			self.V = nn.Conv2d(dim, dim, 1)
			self.proj = nn.Conv2d(dim, dim, 1)

		if self.use_attn:
			self.QK = nn.Conv2d(dim, dim * 2, 1)
			self.attn = WindowAttention(dim, window_size, num_heads)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape

			if w_shape[0] == self.dim * 2:	# QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)
			else:
				gain = (8 * self.network_depth) ** (-1/4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape

		if self.conv_type == 'DWConv' or self.use_attn:
			V = self.V(X)

		if self.use_attn:
			QK = self.QK(X)
			# 把QK和V在维度上进行拼接，方便后续shift window等操作
			QKV = torch.cat([QK, V], dim=1)

			# shift
			shifted_QKV = self.check_size(QKV, self.shift_size > 0)
			Ht, Wt = shifted_QKV.shape[2:]

			# partition windows
			shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
			qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

			attn_windows = self.attn(qkv)

			# merge windows
			shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

			# reverse cyclic shift
			out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
			attn_out = out.permute(0, 3, 1, 2)

			# 如果卷积类型是 Conv 或 DWConv
			# 先通过 self.conv 对前面得到的 V 进行卷积操作。得到 conv_out
			# 然后将 conv_out 和 attn_out 相加，并通过 self.proj 进行投影操作（这里的 self.proj 应该是一个 1×1 卷积层，用于调整特征维度等），得到最终的 out。
			# 如果卷积类型不属于 Conv 或 DWConv，则直接通过 self.proj 对 attn_out 进行投影操作，得到 out。
			if self.conv_type in ['Conv', 'DWConv']:
				conv_out = self.conv(V)
				out = self.proj(conv_out + attn_out)
			else:
				out = self.proj(attn_out)

		else:
			if self.conv_type == 'Conv':
				out = self.conv(X)				# no attention and use conv, no projection
			elif self.conv_type == 'DWConv':
				out = self.proj(self.conv(V))

		return out

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
	return windows

def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x

def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log





class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)


class IDWT(nn.Module):
	def __init__(self):
		super().__init__()
		self.rdwt = SWTInverse(wave='db1', mode='zero', separable=True)

	def forward(self, y):
		b, c, h, w = y.size()
		yl = y[:, 0:c // 4]
		yh = [y[:, c // 4:].reshape(b, 3, c // 4, h, w).permute(0, 2, 1, 3, 4)]
		x = self.rdwt((yl, yh))
		return x

class CIFM(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(CIFM, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		#print(in_feats[0].shape, in_feats[1].shape)
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(self.avg_pool(feats_sum))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias
