import yaml
import torch
from torch import nn

from DWTDemo.DwtFormer import DustFormer, RLN


# 导入DustFormer和相关模块


def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 加载配置文件
    config = load_config('config.yaml')

    # 初始化模型
    model = DustFormer(
        in_chans=config['in_chans'],
        out_chans=config['out_chans'],
        window_size=config['window_size'],
        embed_dims=config['embed_dims'],
        mlp_ratios=config['mlp_ratios'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        attn_ratio=config['attn_ratio'],
        conv_type=config['conv_type'],
        norm_layer=[RLN for _ in config['norm_layer']]
    )

    # 测试数据
    x = torch.randn(1, 3, 256, 256)  # 假设输入图像尺寸为256x256，3个通道
    output = model(x)

    print(f"输入维度: {x.shape}")
    print(f"输出维度: {output.shape}")


if __name__ == "__main__":
    main()
