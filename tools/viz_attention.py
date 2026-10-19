# -*- coding: utf-8 -*-
"""CBAM 注意力图可视化：看训练后的 CBAM 到底在"看"哪里。

为什么做这个（论文/答辩用）
    报告里"CBAM 无稳定增益"目前只有数值结论（mAP50 极差 0.0125、方向随优化器翻转），
    缺机理支撑；而"解决方案"却提出换 SE/ECA —— 同样是注意力模块，说不清 CBAM 为何无效，
    换一个注意力就没有依据。本脚本把训练后 CBAM 内部的**通道注意力向量**和**空间注意力图**
    抓出来，用两个可量化的量回答：
      1) 空间注意力聚焦缺陷，还是背景/边缘？  → 对比度比 = mean(S|框内) / mean(S|框外)
      2) 通道注意力有区分度，还是退化成接近 0.5 的均匀分布？ → 权重 std
    若浅层 SA 对比度比 ≈1 或 <1、CA std 接近 0，就支撑"目标小/样本少/浅层放大背景"这一机理，
    并为"换 SE/ECA 未必有效"提供依据。

实现要点（踩坑记录）
    - CBAM 的中间张量是 forward 内的**局部变量，不落属性**（cbam.py L27-30 / L41-45），
      forward 返回后再也拿不到 → 必须 register_forward_hook 挂在 ca / sa 上捕获。
    - 注入把整层包成 nn.Sequential(原层, CBAM)，故 CBAM 位于 model.model.{i}.1，
      不在 ...cv2.cbam。hook 名字自带层号，一次遍历即可拿到 (层号, 注意力张量)。
    - 带 CBAM 的 best.pt 里 pickle 了 cbam.CBAM 引用，反序列化前必须让 cbam 可 import。
    - letterbox 必须复刻 ultralytics 的 auto=True 分支（dw/dh 先 mod stride）：
      632x480 @ imgsz 320 得到的是 320x256 画布（不是 320x320），差这一步热力图会错位。

用法
  一次看三组（默认，自动抽 8 张测试图聚合）：
    python {ROOT_S}/code/tools/viz_cbam_attention.py
  单组 / 换分辨率（imgsz 默认从该 run 的 args.yaml 自动读）：
    ... viz_cbam_attention.py --models wire_v11_neck_v3 --n-img 20
    ... viz_cbam_attention.py --models wire_v11_640_neck --img firc_xianlan_1082.jpg
  把 CA 图的自适应 y 轴打开（默认固定 [0,1] 便于跨层比较）：
    ... viz_cbam_attention.py --autoscale

输出
    report/_cbam_attention/{run}_spatial.png / _channel.png / summary.txt

结论（2026-10-11 实测：6 个 CBAM 模型 × 21 个 CBAM 模块 × 8 张测试图的**跨图聚合**）
    SA 标准差   中位数 0.0308   最大 0.0960   （[0,1] 尺度）
    CA 标准差   中位数 0.0259
    对比度比    中位数 1.0152   范围 0.747 ~ 1.229（大多数层的跨图极差都横跨 1.0）
    → 空间注意力接近**常数场**、通道门控接近**均匀分布**，两者相乘 ≈ 乘一个常数。

    口径：极差是同一模块在 8 张图上的波动范围，**不是 168 个独立样本**。
    21 个是模块实例数，跨 8 张图各测一次 —— 检验的是"读数稳不稳"，不是"独立抽样"。
    （本句此前误写过"168 次测量"，已按上面的口径改正。）

    机理论证（这是"CBAM 无增益"最硬的解释，不是看图说话）：
      · 通道注意力若近似逐通道常数 → 与后一层 conv 的权重**完全冗余**，
        可以把常数折进 conv 权重，表达能力零增益。
      · 空间注意力若近似逐位置常数（实测 std≈0.03）→ 被后一层 BN 吸收。
        BN 对常数缩放在训练时（batch 统计）与推理时（running 统计）都**严格不变**：
            BN(c·x) = γ(cx − c·μ)/√(c²σ²) + β = γ(x − μ)/σ + β = BN(x)
        故常数缩放的梯度恒为 0，网络**根本无法**从这个自由度上学到东西。
      · 两者叠加 ⇒ CBAM 在功能上接近恒等映射 ⇒ mAP 几乎不动。
    结论在 320/640、v11/v12、backbone/neck/both 上一致，不是分辨率或架构特例。
    残余偏差（如 640 层 4 对比度比 1.11）就是 CBAM 能贡献的全部，量级 1~10%，
    与实验观察到的 ±0.01 mAP 波动同量级 —— 数值与机理自洽。

    注意措辞边界：本结论是"近常数"而非"严格常数"，且只测了 break 类主导的 8 张图，
    论文里写"在测试集样本上接近恒等"，不要写成"CBAM 恒等于恒等映射"。
"""
import argparse
import sys
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 tools/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


# ultralytics 的 best.pt pickle 了 `cbam.CBAM`，反序列化前必须能找到该模块
for _d in (ROOT_S + '/src', ROOT_S + '/src'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cv2
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from cbam import CBAM

RUNS = ROOT_S + '/runs'
IMAGES = ROOT_S + '/dataset/final_bbox/images/test'
LABELS = ROOT_S + '/dataset/final_bbox/labels/test'
OUT = ROOT_S + '/report/_cbam_attention'

NAMES = ['break', 'thunderbolt', 'wear']
PAD_COLOR = (114, 114, 114)
GT_RED = '#c00000'
STRIDE = 32

# 中文字体（与 report/make_report_v4.py 一致）
for _f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc'):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# ------------------------------------------------------------------ 预处理
def letterbox(im, new_shape, stride=STRIDE, auto=True, color=PAD_COLOR):
    """复刻 ultralytics LetterBox(auto=True, center=True)。

    注意 auto=True 下 dw/dh 先对 stride 取模 —— 得到的是"够用就好"的画布，
    632x480 @ 320 会得到 320x256 而非 320x320。漏掉这步热力图就对不上原图。
    """
    shape = im.shape[:2]                      # (H, W)，与 ultralytics 一致
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2                   # center
    if (shape[1], shape[0]) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top, new_unpad[0], new_unpad[1])


def read_gt(label_path, w, h):
    """读 YOLO 格式标签 → 像素 xyxy 列表 [(cls, x1, y1, x2, y2), ...]。"""
    boxes = []
    if not Path(label_path).exists():
        return boxes
    for ln in Path(label_path).read_text(encoding='utf-8').splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c, xc, yc, bw, bh = int(p[0]), *(float(v) for v in p[1:5])
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        boxes.append((c, x1, y1, x2, y2))
    return boxes


def run_imgsz(run):
    """从 run 的 args.yaml 读训练分辨率，读不到则回退 640。"""
    p = Path(RUNS) / run / 'args.yaml'
    if not p.exists():
        return None
    for ln in p.read_text(encoding='utf-8', errors='replace').splitlines():
        if ln.startswith('imgsz:'):
            try:
                return int(ln.split(':', 1)[1].strip())
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------ 抓注意力
def load_model(run):
    from ultralytics import YOLO
    w = Path(RUNS) / run / 'weights' / 'best.pt'
    if not w.exists():
        raise FileNotFoundError(f'缺权重 {w}')
    m = YOLO(str(w))
    m.model.eval()
    return m.model


def locate_cbam(det_model):
    """返回 [(层号, 父层类名, CBAM 实例), ...]，按层号排序。

    判据用**鸭子类型**而非 isinstance(sub, CBAM)：
    `cbam_fixed.CBAM` 是另一个类对象（定义在 cbam_fixed.py），
    isinstance 对它会返回 False，导致修复版模型被判成"没有 CBAM 层"，
    从而**无法用本工具评估修复效果**。这里只认结构：有 .ca 和 .sa。
    """
    seq = det_model.model
    found = []
    for i, layer in enumerate(seq):
        # 注入后被包成 nn.Sequential(原层, CBAM)。
        # 注意先判 nn.Sequential —— ultralytics 的层带 .f 但没有 len()，
        # 直接 len(layer) 会抛 "object of type 'Conv' has no len()"。
        if not (isinstance(layer, nn.Sequential) and len(layer) == 2):
            continue
        sub = layer[1]
        if hasattr(sub, 'ca') and hasattr(sub, 'sa'):
            found.append((i, type(layer[0]).__name__, sub))
    return sorted(found, key=lambda t: t[0])


def capture(det_model, x):
    """一次前向，返回 {层号: (ca向量[C], sa图[H,W], 父层类名)}。"""
    items = locate_cbam(det_model)
    if not items:
        raise RuntimeError('该模型没有 CBAM 层（baseline？）')

    store = {}
    handles = []
    for i, parent, cbam in items:
        def mk(i):
            def hook_ca(_m, _inp, out):
                store.setdefault(i, {})['ca'] = out.detach()[0, :, 0, 0].clone()
            def hook_sa(_m, _inp, out):
                store.setdefault(i, {})['sa'] = out.detach()[0, 0].clone()
            return hook_ca, hook_sa
        hc, hs = mk(i)
        handles.append(cbam.ca.register_forward_hook(hc))
        handles.append(cbam.sa.register_forward_hook(hs))

    try:
        with torch.no_grad():
            det_model(x)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for i, parent, _cbam in items:
        d = store.get(i, {})
        if 'ca' in d and 'sa' in d:
            out[i] = (d['ca'].numpy(), d['sa'].numpy(), parent)
    return out


def map_to_original(sa, canvas_hw, crop_xywh, orig_hw):
    """SA 图 → 原图坐标系：上采样到画布 → 去 letterbox 边 → 缩回原图。"""
    ch, cw = canvas_hw
    left, top, nw, nh = crop_xywh
    full = cv2.resize(sa, (cw, ch), interpolation=cv2.INTER_LINEAR)
    cropped = full[top:top + nh, left:left + nw]
    return cv2.resize(cropped, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)


def contrast_ratio(sa_orig, boxes):
    """缺陷框内 SA 均值 / 框外 SA 均值。>1 看缺陷，≈1 无偏好，<1 看背景。"""
    if not boxes:
        return None
    mask = np.zeros(sa_orig.shape, dtype=bool)
    h, w = sa_orig.shape
    for _c, x1, y1, x2, y2 in boxes:
        xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
        xi2, yi2 = min(int(round(x2)), w), min(int(round(y2)), h)
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
    if not mask.any() or mask.all():
        return None
    return float(sa_orig[mask].mean() / max(sa_orig[~mask].mean(), 1e-8))


# ------------------------------------------------------------------ 绘图
def draw_spatial(run, img_bgr, boxes, per_layer, out_path, alpha, autoscale):
    n = len(per_layer)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.9), squeeze=False, dpi=110)
    axes = axes[0]
    rgb = img_bgr[:, :, ::-1]
    for ax, (i, (ca, sa, parent, crop, r)) in zip(axes, per_layer.items()):
        sa_orig = map_to_original(sa, img_bgr.shape[:2], crop, img_bgr.shape[:2])
        vmin, vmax = (sa_orig.min(), sa_orig.max()) if autoscale else (0.0, 1.0)
        ax.imshow(rgb)
        im = ax.imshow(sa_orig, cmap='jet', alpha=alpha, vmin=vmin, vmax=vmax)
        for c, x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=GT_RED, linewidth=1.6))
            ax.text(x1, max(y1 - 3, 8), NAMES[c] if c < len(NAMES) else str(c),
                    color=GT_RED, fontsize=7.5, fontweight='bold')
        sub = f'对比度比 {r:.3f}' if r is not None else '无 GT 框'
        ax.set_title(f'层 {i} · {parent} · {ca.shape[0]}ch\n'
                     f'SA 均值 {sa.mean():.3f} 标准差 {sa.std():.3f} ｜ {sub}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f'{run} —— CBAM 空间注意力（红框为 GT，jet 越红权重越高）',
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def draw_channel(run, per_layer, out_path, autoscale):
    n = len(per_layer)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.0), squeeze=False, dpi=110)
    axes = axes[0]
    for ax, (i, (ca, _sa, parent, _crop, _r)) in zip(axes, per_layer.items()):
        x = np.arange(len(ca))
        ax.plot(x, ca, lw=0.7, color='#4472c4')
        ax.axhline(0.5, color='#999999', lw=0.8, ls='--')
        ax.axhline(ca.mean(), color=GT_RED, lw=0.9)
        ax.set_title(f'层 {i} · {parent} · {len(ca)}ch\n'
                     f'均值 {ca.mean():.3f} 标准差 {ca.std():.4f}  '
                     f'[{ca.min():.3f}, {ca.max():.3f}]', fontsize=9)
        ax.set_xlabel('通道索引', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('通道注意力权重', fontsize=8)
        if not autoscale:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, ls='--')
        ax.tick_params(labelsize=7)
    fig.suptitle(f'{run} —— CBAM 通道注意力（红线=均值，灰虚线=0.5；标准差越小越接近退化均匀）',
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 主流程
# ------------------------------------------------------------------ 跨图聚合
def list_test_images(prefer_class=0):
    """测试集全部图片，含 prefer_class（默认 break）的排在前面。

    注意力图是输入相关的，只测一张下不了普遍结论 —— 默认多张聚合。
    """
    pref, other = [], []
    for p in sorted(Path(IMAGES).glob('*.jpg')):
        lb = Path(LABELS) / f'{p.stem}.txt'
        cls = set()
        if lb.exists():
            for ln in lb.read_text(encoding='utf-8').splitlines():
                tok = ln.split()
                if tok:
                    cls.add(int(tok[0]))
        (pref if prefer_class in cls else other).append(p.name)
    return pref + other


def fmt_mm(vals, nd=4):
    """跨图聚合：均值 [最小~最大]；无有效值给 -。"""
    if not vals:
        return '-'
    return f'{np.mean(vals):.{nd}f} [{min(vals):.{nd}f}~{max(vals):.{nd}f}]'


def process_model(run, img_names, imgsz_arg, out_dir, alpha, autoscale):
    """一个模型只加载一次，跑完所有图；图只画第一张，指标跨图聚合。"""
    det = load_model(run)
    imgsz = imgsz_arg or run_imgsz(run) or 640

    agg, first, used = {}, None, 0
    for name in img_names:
        img_bgr = cv2.imread(str(Path(IMAGES) / name))
        if img_bgr is None:
            print(f'  !! 跳过读取失败的图 {name}')
            continue
        h, w = img_bgr.shape[:2]
        boxes = read_gt(Path(LABELS) / f'{Path(name).stem}.txt', w, h)

        lb, _r, crop = letterbox(img_bgr, imgsz)
        canvas_hw = lb.shape[:2]
        x = torch.from_numpy(
            np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
        ).float().unsqueeze(0) / 255.0

        raw = capture(det, x)
        per_layer = {}
        for i, (ca, sa, parent) in raw.items():
            sa_orig = map_to_original(sa, canvas_hw, crop, (h, w))
            ratio = contrast_ratio(sa_orig, boxes)
            per_layer[i] = (ca, sa, parent, crop, ratio)
            a = agg.setdefault(i, {'parent': parent, 'ch': int(ca.shape[0]),
                                   'ca_std': [], 'ca_mean': [], 'sa_mean': [],
                                   'sa_std': [], 'ratio': []})
            a['ca_std'].append(float(ca.std()))
            a['ca_mean'].append(float(ca.mean()))
            a['sa_mean'].append(float(sa.mean()))
            a['sa_std'].append(float(sa.std()))
            if ratio is not None:
                a['ratio'].append(float(ratio))
        used += 1
        if first is None:
            first = (name, img_bgr, boxes, crop, canvas_hw, per_layer)

    if used == 0:
        raise RuntimeError('没有任何图成功处理')
    draw_spatial(run, first[1], first[2], first[5],
                 Path(out_dir) / f'{run}_spatial.png', alpha, autoscale)
    draw_channel(run, first[5], Path(out_dir) / f'{run}_channel.png', autoscale)
    return agg, first, imgsz, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['wire_v11_backbone_v3', 'wire_v11_neck_v3',
                             'wire_v11_both_v3'],
                    help='runs/ 下的 run 名，可多个')
    ap.add_argument('--img', nargs='*', default=[],
                    help='指定测试图文件名；不给则按 --n-img 自动抽（优先含 break）')
    ap.add_argument('--n-img', type=int, default=8,
                    help='自动抽图张数（--img 给了则忽略）')
    ap.add_argument('--imgsz', type=int, default=0,
                    help='0=从各 run 的 args.yaml 自动读')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--alpha', type=float, default=0.5, help='热力图叠加透明度')
    ap.add_argument('--autoscale', action='store_true',
                    help='热力图/通道图按各自实测范围取色（默认固定 [0,1] 便于跨层比较）')
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    imgs = args.img if args.img else list_test_images()[:args.n_img]

    hdr = (f'{"层":>4}{"父层":>8}{"通道":>6}{"CA标准差(跨图)":>27}'
           f'{"SA标准差(跨图)":>27}{"对比度比(跨图)":>27}{"图数":>6}')
    lines = ['=' * 100,
             'CBAM 注意力可视化（指标为多张测试图的跨图聚合）',
             f'模型 {", ".join(args.models)}',
             f'用图 {len(imgs)} 张：{" ".join(imgs)}',
             '',
             'SA 对比度比 = mean(S|GT框内) / mean(S|框外)；>1 看缺陷，≈1 无偏好，<1 看背景',
             'SA 标准差衡量"图本身有没有结构"：标准差小=接近常数图（乘常数会被下一层 BN 吸收，等效恒等）',
             'CA 标准差越小 → 通道权重越接近退化的均匀分布（无区分度）',
             '注 1：SA 作用于通道注意力加权后的张量（CBAM 内部顺序为 ca → sa）',
             '注 2：注意力是输入相关的，单图结论不可推广 —— 本表跨图取均值并给出极差',
             '注 3：PNG 只画第一张图用于定性观察，表为全部图的聚合',
             '=' * 100]
    all_ratio, all_sa_std, all_ca_std = [], [], []

    for run in args.models:
        try:
            agg, first, imgsz, used = process_model(
                run, imgs, args.imgsz, args.out, args.alpha, args.autoscale)
        except Exception as e:
            print(f'!! {run}: {type(e).__name__}: {e}')
            lines += ['', f'!! {run}: {type(e).__name__}: {e}']
            continue

        name0, img_bgr, boxes, crop, canvas_hw, _pl = first
        print(f'\n=== {run} ===   imgsz={imgsz}   成功 {used}/{len(imgs)} 张')
        print(f'  画图用 {name0}  {img_bgr.shape[1]}x{img_bgr.shape[0]}'
              f'   letterbox 画布 {canvas_hw[1]}x{canvas_hw[0]}'
              f'   padding(left,top)={crop[:2]}')
        print(hdr)
        for i, a in sorted(agg.items()):
            print(f'{i:>4}{a["parent"]:>8}{a["ch"]:>6}'
                  f'{fmt_mm(a["ca_std"]):>27}{fmt_mm(a["sa_std"]):>27}'
                  f'{fmt_mm(a["ratio"], 3):>27}{len(a["sa_std"]):>6}')
            all_ca_std += a['ca_std']
            all_sa_std += a['sa_std']
            all_ratio += a['ratio']

        lines += ['', '-' * 100,
                  f'{run}   imgsz={imgsz}   成功 {used}/{len(imgs)} 张   画图用 {name0}'
                  f'   letterbox 画布 {canvas_hw[1]}x{canvas_hw[0]}', hdr]
        for i, a in sorted(agg.items()):
            lines.append(f'{i:>4}{a["parent"]:>8}{a["ch"]:>6}'
                         f'{fmt_mm(a["ca_std"]):>27}{fmt_mm(a["sa_std"]):>27}'
                         f'{fmt_mm(a["ratio"], 3):>27}{len(a["sa_std"]):>6}')

    if all_ratio:
        lines += ['', '=' * 100, '总体（全部模型 × 全部层 × 全部图）',
                  f'  SA 标准差   中位数 {np.median(all_sa_std):.4f}'
                  f'   最大 {max(all_sa_std):.4f}   样本 {len(all_sa_std)}',
                  f'  CA 标准差   中位数 {np.median(all_ca_std):.4f}'
                  f'   最大 {max(all_ca_std):.4f}',
                  f'  对比度比    中位数 {np.median(all_ratio):.4f}'
                  f'   范围 {min(all_ratio):.3f} ~ {max(all_ratio):.3f}',
                  '=' * 100]
        print(f'\n总体：SA 标准差中位数 {np.median(all_sa_std):.4f}'
              f'（最大 {max(all_sa_std):.4f}）   '
              f'CA 标准差中位数 {np.median(all_ca_std):.4f}   '
              f'对比度比中位数 {np.median(all_ratio):.4f}'
              f'（{min(all_ratio):.3f}~{max(all_ratio):.3f}）')

    Path(args.out, 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {args.out}/summary.txt')


if __name__ == '__main__':
    main()
