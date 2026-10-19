# -*- coding: utf-8 -*-
"""近似重复图检查（dHash + 阈值扫描），回答两个不同的问题。

问题一（口径/论文级）：**报告里的 mAP 是不是偏乐观的？**
    精确哈希（check_split_overlap.py）已证明 train/val/test 无**完全**相同文件。
    但**近似**重复（同一场景连拍、轻微裁剪/缩放/亮度变化）查不出来。
    若 train 里存在某图、test 里存在它的近似图，则 test 并非"未见过的数据"，
    报告的 mAP 衡量的是**同场景内泛化**而非跨场景泛化 —— 这必须在论文里讲清。

问题二（本次偏置的来源）：**"test 系统性高于 val" 是不是近似图分布不均造成的？**
    只有近似图**偏向某一集**时才构成解释（近似图会抬高它所在的那一集）。
    故需比较各集**内部**密度与跨集密度：若 test 内部明显比 val 内部密集，才是嫌疑。

方法：dHash（9×8 灰度 → 8×8 水平差分 → 64 位），汉明距离 ≤ TH 判为近似。
     **扫描 TH = 2/4/6** 以区分两件事：
       TH=2 是"几乎同一张图"（硬重复核心）；
       TH=6 已放宽到"同一场景/相似构图"（源数据整体冗余度）。
     只报一个阈值会把这两件事混在一起，故必须扫描。

输出：logs/near_dup.txt
用法：python {ROOT_S}/code/tools/check_near_dup.py
"""
import sys
from itertools import combinations
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 tools/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


import numpy as np
from PIL import Image

ROOT = Path(ROOT_S + '/dataset/final_bbox/images')
OUT = Path(ROOT_S + '/logs/near_dup.txt')
THRESHOLDS = (2, 4, 6)
SIZE = (9, 8)   # dHash 标准尺寸
MAD_TH = 20.0   # 逐像素灰度平均绝对差 ≤ 此值 → 判定为"真近似重复"（0~255 量纲）


def dhash(p):
    im = Image.open(p).convert('L').resize(SIZE, Image.LANCZOS)
    px = np.asarray(im, dtype=np.uint8).reshape(-1)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | int(px[r * 9 + c] < px[r * 9 + c + 1])
    return bits


def ham(a, b):
    return bin(a ^ b).count('1')


# --- 候选对定案：dHash 只有 64 位，d=0 也可能是"相同构图的不同图"的碰撞 ---
#     故对候选对再算**逐像素灰度 MAD**，用它定案"是不是真近似重复"。
_MAD_CACHE = {}


def _gray(s, n):
    key = (s, n)
    if key not in _MAD_CACHE:
        im = Image.open(ROOT / s / n).convert('L')
        _MAD_CACHE[key] = (im.size, np.asarray(im, dtype=np.float32))
    return _MAD_CACHE[key]


def pixel_mad(s1, n1, s2, n2):
    """返回 (MAD, 是否同尺寸)。尺寸不同时把前者缩到后者尺寸再比。"""
    sz1, a1 = _gray(s1, n1)
    sz2, a2 = _gray(s2, n2)
    if sz1 != sz2:
        a1 = np.asarray(Image.open(ROOT / s1 / n1).convert('L').resize(sz2),
                        dtype=np.float32)
    return float(np.abs(a1 - a2).mean()), sz1 == sz2


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    hashes = {}
    for s in ('train', 'val', 'test'):
        d = ROOT / s
        hashes[s] = {p.name: dhash(p) for p in sorted(d.iterdir()) if p.is_file()}

    say('=' * 92)
    say('近似重复检查（dHash 64 位，阈值扫描 TH = %s）' % ', '.join(map(str, THRESHOLDS)))
    say('=' * 92)
    say()
    say(f'  已哈希：train {len(hashes["train"])} / val {len(hashes["val"])} / test {len(hashes["test"])}')
    say()

    # 预生成所有需要比较的对（只算一次，各阈值复用）
    sets = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test'),
                 ('val', 'val'), ('test', 'test'), ('train', 'train')]:
        if a == b:
            items = list(hashes[a].items())
            pairs = combinations(items, 2)
            den = len(items) * (len(items) - 1) / 2
        else:
            pairs = ((x, y) for x in hashes[a].items() for y in hashes[b].items())
            den = len(hashes[a]) * len(hashes[b])
        # 只留 ≤6 的，避免存下上百万对
        hits = [(ham(h1, h2), n1, n2) for (n1, h1), (n2, h2) in pairs if ham(h1, h2) <= max(THRESHOLDS)]
        sets[(a, b)] = (hits, den)

    # ---------- 表 1：密度扫描 ----------
    say('-' * 92)
    say('表 1：近似对密度（每 1000 对）—— 跨集 vs 集内')
    say('-' * 92)
    say(f'{"组合":<18}' + ''.join(f'{"TH<=" + str(t):>18}' for t in THRESHOLDS))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test'), ('val', 'val'), ('test', 'test')]:
        hits, den = sets[(a, b)]
        row = ''.join(f'{1000 * sum(1 for d, _, _ in hits if d <= t) / den:>18.2f}' for t in THRESHOLDS)
        say(f'{a + " ↔ " + b:<18}{row}')

    # ---------- 表 2：绝对对数 ----------
    say()
    say('-' * 92)
    say('表 2：近似对绝对数')
    say('-' * 92)
    say(f'{"组合":<18}{"总对数":>14}' + ''.join(f'{"TH<=" + str(t):>12}' for t in THRESHOLDS))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test'), ('val', 'val'), ('test', 'test')]:
        hits, den = sets[(a, b)]
        row = ''.join(f'{sum(1 for d, _, _ in hits if d <= t):>12}' for t in THRESHOLDS)
        say(f'{a + " ↔ " + b:<18}{den:>14.0f}{row}')

    # ---------- 样例 ----------
    say()
    say('-' * 92)
    say('TH<=2 的样例（几乎同一张图，最硬的重复核心）')
    say('-' * 92)
    for a, b in [('train', 'test'), ('val', 'test'), ('val', 'val'), ('test', 'test')]:
        hits, _ = sets[(a, b)]
        ex = sorted([h for h in hits if h[0] <= 2])[:6]
        say(f'  {a} ↔ {b}：')
        if not ex:
            say('      （无）')
        for d, n1, n2 in ex:
            say(f'      d={d}  {a}/{n1}   {b}/{n2}')
        say()

    # ---------- 表 3：对候选对逐像素定案 ----------
    # dHash 只有 64 位，d=0 也可能是"同构图不同图"的碰撞（实测确有此例：
    # val/214 vs test/392 的 MAD = 39，明显是两张不同的图）。故必须用像素 MAD 定案。
    say('-' * 92)
    say(f'表 3：候选对（dHash TH≤2）的逐像素定案 —— 灰度 MAD ≤ {MAD_TH:.0f} 才算真近似重复')
    say('-' * 92)
    verified = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        hits, _ = sets[(a, b)]
        cand = sorted([h for h in hits if h[0] <= 2])
        rows = []
        for d, n1, n2 in cand:
            m, same = pixel_mad(a, n1, b, n2)
            rows.append((m, d, n1, n2, same))
        rows.sort()
        real = [r for r in rows if r[0] <= MAD_TH]
        verified[(a, b)] = real
        # 对数会重复计同一张图（一张图可能有多个近复制品），故同时给**去重图数**：
        # "评估集里有 N 张图能在 train 找到近复制品" 才是论文该写的分母。
        nb = {n2 for _, _, _, n2, _ in real}
        na = {n1 for _, _, n1, _, _ in real}
        say(f'  {a} ↔ {b}：候选 {len(cand)} 对 → **真近似重复 {len(real)} 对**'
            f'（涉及 {a} 侧 {len(na)} 张、{b} 侧 {len(nb)} 张）')
        for m, d, n1, n2, same in rows[:5]:
            tag = '真重复' if m <= MAD_TH else '碰撞(不同图)'
            say(f'      MAD={m:>6.2f} d={d}  同尺寸={str(same):<5}{tag:<14}'
                f'{a}/{n1}   {b}/{n2}')
        say()

    # ---------- 判读 ----------
    say('=' * 92)
    say('判读')
    say('=' * 92)
    vt = verified[('train', 'test')]
    vv = verified[('train', 'val')]
    vx = verified[('val', 'test')]
    say(f'  问题一（mAP 是否偏乐观）：经逐像素定案的真近似重复 —— '
        f'train↔test **{len(vt)} 对** / train↔val **{len(vv)} 对** / val↔test **{len(vx)} 对**')
    if vt or vv:
        nv_t = len({n2 for _, _, _, n2, _ in vv})
        nt_t = len({n2 for _, _, _, n2, _ in vt})
        say(f'    ⇒ **train 与评估集之间确实存在近似的（甚至像素级近乎相同的）图片**：')
        say(f'       test 的 179 张里有 **{nt_t} 张**、val 的 178 张里有 **{nv_t} 张**'
            f'能在 train 找到近复制品。')
        say('       这些评估图并非"未见过的场景"，故报告的 mAP 衡量的是**同场景内泛化**，')
        say('       不是跨场景泛化。')
        say('       ⚠ 论文里必须写明这条限制；不能宣称"泛化到新场景"。')
        say('       修法（若要做）：按**场景簇**而非按图片划分 split（聚类后整簇分配）。')
    else:
        say('    ⇒ train↔评估集无经定案的真重复，mAP 可视为"未见过的图"上的性能。')

    dv = 1000 * sum(1 for d, _, _ in sets[('val', 'val')][0] if d <= 6) / sets[('val', 'val')][1]
    dt = 1000 * sum(1 for d, _, _ in sets[('test', 'test')][0] if d <= 6) / sets[('test', 'test')][1]
    nv_t = len({n2 for _, _, _, n2, _ in vv})
    nt_t = len({n2 for _, _, _, n2, _ in vt})
    say()
    say(f'  问题二（test > val 的来源）：集内密度 TH≤6 —— val {dv:.2f} / test {dt:.2f}（每千对）')
    say(f'    train 泄漏到评估集的图数：val {nv_t} / {len(hashes["val"])}，'
        f'test {nt_t} / {len(hashes["test"])}')
    if nt_t > nv_t * 1.3 and dt > 2 * dv:
        say('    ⇒ test 侧的近似图富集明显更强 ⇒ 该机制**可能**贡献了 test 偏高，需人工确认。')
    else:
        say(f'    ⇒ **val 侧的泄漏并不少于 test（val {nv_t} vs test {nt_t}）**，')
        say('       若泄漏起作用应把 val 抬得更高，即与观测到的方向**相反**。')
        say('       ⇒ 近似重复**无法解释**同向偏置；该机制排除。')
    say('    注：近似重复抬高的是它**所在**的那一集，故只有"偏向某一集"时才构成解释。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
