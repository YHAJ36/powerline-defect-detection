# -*- coding: utf-8 -*-
"""标准 CBAM 注意力模块 + 向 YOLO 模型注入的函数。

CBAM = 通道注意力（Channel Attention）+ 空间注意力（Spatial Attention）。
注入方式：把目标层包装成 nn.Sequential(原层, CBAM)，不改变通道数、不改变层索引。
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：全局平均池化 + 全局最大池化 → 共享 MLP → sigmoid。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享 MLP（两个 1x1 卷积）
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """空间注意力：通道维度平均池化 + 最大池化 → 拼接 → 7×7 卷积 → sigmoid。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(y))


class CBAM(nn.Module):
    """CBAM：先通道注意力，再空间注意力。输入输出同形状。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


def _get_out_channels(m):
    """获取模块的输出通道数。"""
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        if hasattr(conv, 'out_channels'):
            return conv.out_channels
        if hasattr(conv, 'conv') and isinstance(conv.conv, nn.Conv2d):
            return conv.conv.out_channels
    # 兜底：模块内最后一个 Conv2d 的输出通道
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('无法确定模块 %s 的输出通道数' % type(m).__name__)


def _get_sequential(model):
    """从 YOLO 对象 / DetectionModel / nn.Sequential 中取出主干 Sequential。"""
    if isinstance(model, nn.Sequential):
        return model
    if hasattr(model, 'model'):
        inner = model.model
        if isinstance(inner, nn.Sequential):
            return inner
        # model 是 YOLO wrapper，其 .model 是 DetectionModel
        if hasattr(inner, 'model') and isinstance(inner.model, nn.Sequential):
            return inner.model
    raise TypeError('无法识别模型结构，无法定位 Sequential 层序列')


def inject_cbam(model, target_names, mode='all', reduction=16):
    """在指定类型的模块后插入 CBAM。

    参数:
        model: YOLO 对象 / DetectionModel / nn.Sequential 均可（自动定位层序列）
        target_names: 目标模块类名集合，如 {'C3k2'} 或 {'A2C2f'}
        mode: 'all' 全部插入；'backbone_last3' 只在 backbone 最后三层插入
        reduction: CBAM 通道降维比

    返回:
        插入 CBAM 的层索引列表
    """
    seq = _get_sequential(model)  # nn.Sequential

    # backbone 边界：第一个 Upsample 之前为 backbone
    upsample_idx = None
    for i, m in enumerate(seq):
        if isinstance(m, nn.Upsample):
            upsample_idx = i
            break

    indices = [i for i, m in enumerate(seq) if type(m).__name__ in target_names]

    if mode == 'backbone_last3':
        indices = [i for i in indices if (upsample_idx is None or i < upsample_idx)]
        indices = indices[-3:]
        note = '（简化：仅 backbone 最后 3 层）'
    else:
        note = '（全部 %d 层）' % len(indices)

    for i in indices:
        ch = _get_out_channels(seq[i])
        cbam = CBAM(ch, reduction=reduction)
        wrapper = nn.Sequential(seq[i], cbam)
        # 保留 forward 路由所需的 f（来源索引）和 i（自身索引）属性
        wrapper.f = seq[i].f
        wrapper.i = seq[i].i
        if hasattr(seq[i], 'type'):
            wrapper.type = seq[i].type
        seq[i] = wrapper

    print('已注入 CBAM：目标类型=%s 模式=%s 插入层索引=%s'
          % (sorted(target_names), note, indices))
    return indices


if __name__ == '__main__':
    # 自测：打印模块结构
    print('ChannelAttention 模块:')
    ca = ChannelAttention(256)
    print(ca)
    print('\nSpatialAttention 模块:')
    sa = SpatialAttention(7)
    print(sa)
    print('\nCBAM 模块:')
    cbam = CBAM(256)
    print(cbam)
    # 前向自测
    x = torch.randn(2, 256, 20, 20)
    y = cbam(x)
    print('\n输入形状:', tuple(x.shape), '→ 输出形状:', tuple(y.shape), '(应一致)')
