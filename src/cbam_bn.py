# -*- coding: utf-8 -*-
"""CBAM+BN —— 只给 CBAM 的空间注意力读出头加一层 BatchNorm。

为什么有这个文件（2026-10-15，接 cbam_fixed 失败之后）
    报告第五章定量证明 CBAM 在本数据集近似恒等映射，根因定位到 **SA 读出头**
    （`SpatialAttention` 的 7×7 conv + sigmoid）。两个机制都由 sigmoid 造成：
      （一）**DC 饱和** —— 正均值输入 × 无偏置 conv ⇒ sigmoid 前 z 出现大正 DC
           （训练后实测层 2/4/6 = +4.94/+3.15/+4.41），单边饱和成常数图；
      （二）**幅度衰减** —— 层 8/9 未饱和（z 均值 −0.23/−0.08），但 conv 输出幅度
           过小，sigmoid 在其线性区把幅度再压 4 倍 ⇒ 输出 std 仅 0.0307/0.0306，
           塌成常数 0.5。
    两者的共同根因是 **没有任何东西约束 sigmoid 的输入 z**：尺度与偏移全由 conv
    随机初值决定，而饱和区梯度 ≈0，训练修不动它。

    第一版修复 `cbam_fixed.py`（SA conv 加 bias + 输入空间中心化）**已失败**：
    跑满 60 轮，DC 饱和幅度降到基线的 44%/20%，但 **mAP 没涨**（val 0.8810 < 0.8861）。
    机理上 DC 只是被「改道」进 bias —— bias 是个自由参数，没有任何机制阻止它长大。

本文件的修复（**只改一处**）
    SA 读出头从        z = conv(cat[mean_c(x), max_c(x)]);  s = sigmoid(z)
    改为               z = BN(conv(cat[mean_c(x), max_c(x)]));  s = sigmoid(z)
    BN 减 batch 统计均值（**消 DC**）、除 batch 统计标准差（**钉尺度**），
    两个机制同时切断。这与 `triplet_att.AttentionGate` 的做法**同构**
    （Triplet 的 gate 就是 `sigmoid(BN(conv(ZPool(x))))`），
    差别只在注意力分支的组织方式（Triplet 三分支轮换维度、CBAM 通道+空间串联）。

    与 `cbam_fixed.py` 的关键区别：cbam_fixed 的两处改动都**不消除** DC 的产生源
    （conv 仍可学到任意大的 bias）；本文件让 BN 在**每次前向**里把 DC 减掉，
    且 BN 的 scale 由数据决定，不继承 conv 幅度 —— DC 无处可藏。

    ⚠ BN 只在**训练态**（或 track_running_stats=False）用 batch 统计；
      推理态用 running 统计（训练中累积）。故"未训练前向闸门"必须**在 train() 态**
      下测，否则未训练的 BN（running_mean=0, running_var=1）近似恒等，测不出效果。
      见 `code/tools/check_cbam_bn.py`。

参数量
    相对原版 CBAM，每个 SA 多一个 `BatchNorm2d(1, affine=True)` = weight(1) + bias(1)
    = **+2 参数/层**（running_mean/running_var 是 buffer，不计入 parameters()）。
    这是"只改一处"的直接代价，量级可忽略，不构成容量差异的借口。

保留不变（保证"只改一项"）
    · `ChannelAttention` 一字未动（仍 bias=False 的两层 1×1 MLP）；
    · `CBAM.forward` 仍是 `x = x*ca(x); x = x*sa(x)`；
    · SA 的 conv 仍 `bias=False`（BCN 自带的 affine 提供偏移，再加 conv bias 是第二处改动）。
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：全局平均池化 + 全局最大池化 → 共享 MLP → sigmoid。（与 cbam.py 逐字相同）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
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
    """空间注意力 + **读出头 BN** —— 本文件相对 cbam.py 的唯一实质改动。

    BN 超参与 triplet_att.AttentionGate 的 BasicConv 保持一致
    （eps=1e-5, momentum=0.01, affine=True），便于两臂互为印证时口径统一。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        # ★ 新增：sigmoid 之前的归一化，消 DC + 钉尺度
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.bn(self.conv(y)))    # ← 原 cbam.py: sigmoid(self.conv(y))


class CBAM(nn.Module):
    """CBAM+BN：先通道注意力，再空间注意力（含读出头 BN）。输入输出同形状。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


if __name__ == '__main__':
    # 前向自测：形状不变 + BN 在 train 态确实把 z 的均值压到 0
    sa = SpatialAttention()
    sa.train()
    x = torch.randn(4, 256, 20, 20) + 3.0        # 故意给正均值输入（模拟 DC 来源）
    z = sa.bn(sa.conv(torch.cat([x.mean(1, keepdim=True),
                                 x.max(1, keepdim=True)[0]], dim=1)))
    print('SA 输出 std（train 态）:', float(sa(x).std()))
    print('sigmoid 前 z 均值:', float(z.mean()), '  z std:', float(z.std()))

    cbam = CBAM(256)
    y = cbam(torch.randn(2, 256, 20, 20))
    print('CBAM+BN 形状:', tuple(y.shape), '(应与输入一致)')
