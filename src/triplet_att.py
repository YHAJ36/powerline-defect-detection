# -*- coding: utf-8 -*-
"""Triplet Attention —— 官方实现复刻（Rotate to Attend: Convolutional Triplet Attention Module, WACV 2021）。

来源
    官方仓库 LandskapeAI/triplet-attention，文件 `triplet_attention.py`（仓库根目录）。
    取源途径：raw.githubusercontent.com 本机不通（HTTP 000），经 ghproxy.net 镜像取得原文。
    本文件**逐字复刻**其模块定义（BasicConv / ZPool / AttentionGate / TripletAttention），
    只做三件非语义改动：① 加中文注释与出处；② 显式写出 `import`；③ 无。

为什么本文件的存在值得单独写一段（2026-10-14 审查结论）
    CBAM 在本数据集被定量证明失效，两个机制：
      （一）**DC 饱和** —— 正均值输入 × 无偏置 7×7 conv ⇒ sigmoid 前 z 出现大正 DC
           （训练后实测层 2/4/6 = +4.94/+3.15/+4.41），单边饱和成常数图。
      （二）**幅度衰减** —— 层 8/9 未饱和（z 均值 −0.23/−0.08），但输出 std 仅
           0.0307/0.0306、框内外比 0.98/0.99 ⇒ 塌成常数、无区分度。
    两个机制的共同根因是**没有任何东西约束 sigmoid 的输入**：尺度与偏移全由 conv 的
    随机初值决定，而饱和区梯度 ≈0，训练修不动它。

    Triplet 的 `AttentionGate` 在 conv 与 sigmoid 之间**多了一层 `BatchNorm2d`**：
        z = BN(conv(ZPool(x)))     ← 这一层是 CBAM 没有的
        scale = sigmoid(z)
    BN 减去 batch 均值（消 DC）、除以 batch 标准差（钉尺度），**两个机制同时被切断**。
    实测证据（`code/upscale/probe_triplet_dc.py` 与 `probe_triplet_vs_cbam.py`）：
      · train 态下 sigmoid 前的 z 均值恒为 +0.0000±0.0000（5 种子无一例外）；
      · 五层 scale 标准差 0.178~0.206，等于 z~N(0,1) 时 std(sigmoid(z))=0.2082 的理论值
        （即这个尺度由 BN 决定，不继承 conv 幅度）；对照 CBAM 训练后仅 0.032。

    ⚠ 但这**只说明 Triplet 不会像 CBAM 那样退化成常数**，**不说明它有区分度**。
      区分度必须训练后测。本实验定性为**排除性实验，非提升实验**（收益预期 <1pt）。

注入注意（参照 CBAM 的失效分析结论）
    CBAM 曾因"在 model.train() 之前注入"被 `train()` 内部的 yaml 重建丢弃，
    静默失效。正确做法是**子类化 `DetectionTrainer.get_model()`**，在重建后立即注入。
    且注入后**必须做参数量差检验**（`code/tools/check_triplet_params.py`）：
    参数量没变 = 注入又被丢了。本模块单个 TripletAttention 的可训练参数见该脚本预期值。
"""
import torch
import torch.nn as nn


class BasicConv(nn.Module):
    """官方 BasicConv。注意 bias 默认 False、bn 默认 True —— 这正是切片位置。"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    """跨通道的 max 与 mean 拼接 → (B,2,H,W)。"""

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    """官方 AttentionGate：conv(无 bias) → BN → sigmoid，作用于 x。"""

    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    """官方 TripletAttention：三个分支（C×W / H×C / H×W）等权平均。

    输入输出同形状（B,C,H,W），可即插即用（与 C3k2 / SPPF 兼容）。
    """

    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
