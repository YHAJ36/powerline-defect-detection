# -*- coding: utf-8 -*-
"""修复版 CBAM 注意力模块 —— 针对「读出头结构性缺陷」的两处改动。

背景（见 archive/2026-10-11.md 与 code/tools/analyze_cbam_degradation.py）：
标准 cbam.py 的空间注意力在 wire_v11_640_both 上退化成近常数图，定量证据：

  层   z 均值(DC)   z 均值绝对值   输出 std   输出框内外比
  2      +4.94        4.94         0.0159      1.012     ← 单边饱和到常数 1
  4      +3.15        3.15         0.0823      1.085
  6      +4.41        4.41         0.0390      1.028
  8      -0.23        0.23         0.0307      0.981     ← 线性区、输出 ≈ 常数 0.5
  9      -0.08        0.08         0.0306      0.995

（"z 均值 == |z| 均值"说明层 2/4/6 的 z 几乎处处为正；更关键的是**噪声图的 |z| 反而更大**
  8.31 / 4.98 / 6.71 ⇒ 这个 DC 与图像内容无关，是"正均值输入 × 无偏置 conv"的常数偏置。）

本文件相对 cbam.py 只改 SpatialAttention 两处，ChannelAttention 与 CBAM 逻辑不变：

  1. conv 加 bias=True —— 给网络一个**专门用来抵消 DC 的参数**。
     原版 bias=False，DC 只能靠 7×7 卷积核的随机初值间接产生，而梯度里没有
     修正它的方向（饱和区 sigmoid 导数 ≈ 0），于是 DC 停在初始随机游走的取值上。
  2. forward 里先做**空间中心化** y = y - y.mean(dim=(2,3), keepdim=True) ——
     把输入拉回零均值，sigmoid 就落在线性区，不再单边饱和。

二者叠加后的期望（未训练前向即可验证）：SA 输出 std 应显著大于 0.05、
框内外比应大于 1.1（即真的对缺陷有区分度）。这由 code/tools/check_cbam_fixed.py 的闸门检验。
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：全局平均池化 + 全局最大池化 → 共享 MLP → sigmoid。（与 cbam.py 相同）"""

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
    """空间注意力（修复版）：中心化的通道平均/最大池化 → 拼接 → 7×7 卷积（带 bias）→ sigmoid。

    与 cbam.py 的差异只有两处，见文件头。改动很小，但直接针对已定位的失效机制。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        # 改动 1：bias=True。原版 bias=False 时，抵消输入 DC 只能靠卷积核的随机初值，
        # 而饱和区梯度 ≈ 0，loss 无法把这个 DC 拉回来。
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        # 改动 2：空间中心化。输入的两个通道（通道均值图 / 通道最大值图）都对
        # 整张特征图有正偏置，叠加卷积的正响应后 z 出现 +3~+5 的大正 DC → sigmoid 饱和。
        # 减掉每张图自身的空间均值后，z 回到 0 附近，落在 sigmoid 的线性区。
        y = y - y.mean(dim=(2, 3), keepdim=True)
        return self.sigmoid(self.conv(y))


class CBAM(nn.Module):
    """CBAM：先通道注意力，再空间注意力。输入输出同形状。（与 cbam.py 相同）"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


if __name__ == '__main__':
    print('SpatialAttention（修复版）:')
    print(SpatialAttention(7))
    # 前向自测：模拟"正均值输入"（原版正是在这种输入上饱和）
    sa = SpatialAttention(7)
    x = torch.rand(2, 64, 20, 20) * 2.0
    with torch.no_grad():
        y = sa(x)
    print('\n输入 %s → 输出 %s' % (tuple(x.shape), tuple(y.shape)))
    print('输出 min/max/std = %.4f / %.4f / %.4f' % (y.min(), y.max(), y.std()))
    print('（原版在这种"正均值输入"上 std 会小到 0.03 量级）')
