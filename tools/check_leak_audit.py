# -*- coding: utf-8 -*-
"""leak/clean 划分的一致性审计 —— 检查 make_leak_subsets.py 有没有漏检或错分。

为什么必须做：分级统计显示，`clean` 子集里有 11 张图
与 train 的最近 **dHash** 距离 ≤2。若这 11 张同时满足 **MAD≤20**，那它们本该被划进 leak
—— 即 P0 的 "clean" 里混着近复制品，**泄漏量化本身的输入就是错的**，必须重做。
若它们全被 MAD 闸挡下（dHash 碰撞，实际不是同一张），则划分自洽，P0 成立。

⚠ 两个口径必须分清（否则会误判）：
   分级剖面（subset_distribution.py）只用 **dHash**；
   leak 定义是 **dHash≤2 且 MAD≤20** 两道闸。故 clean 里出现 dHash≤2 的图**不一定**是漏检。

做法：对每张评估图，遍历 train 取 dHash 汉明距离 ≤2 的候选，逐个算全分辨率灰度 MAD
（口径与 check_near_dup.py 完全一致：SIZE=(9,8) dHash；MAD 在图原始分辨率上算，
尺寸不同则把前者缩到后者尺寸），看是否存在 MAD≤20 的候选。
  - `leak` 里的图：应**都有** (否则 leak 定义自己不自洽)
  - `clean` 里的图：应**都没有** (否则 clean 被污染)

只读，纯 CPU。
输出：logs/leak_audit.txt
用法：python {ROOT_S}/code/tools/check_leak_audit.py
"""
import sys
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 tools/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


import numpy as np
from PIL import Image

ROOT = Path(ROOT_S + '/dataset/final_bbox/images')
SUB = Path(ROOT_S + '/dataset/leak_subsets')
OUT = Path(ROOT_S + '/logs/leak_audit.txt')
SIZE = (9, 8)
MAD_TH = 20.0
HAM_TH = 2


def dhash(p):
    im = Image.open(p).convert('L').resize(SIZE, Image.LANCZOS)
    px = np.asarray(im, dtype=np.uint8).reshape(-1)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | int(px[r * 9 + c] < px[r * 9 + c + 1])
    return bits


_CACHE = {}


def gray(p):
    if p not in _CACHE:
        im = Image.open(p).convert('L')
        _CACHE[p] = (im.size, np.asarray(im, dtype=np.float32))
    return _CACHE[p]


def mad(p1, p2):
    (s1, a1), (s2, a2) = gray(p1), gray(p2)
    if s1 != s2:
        a1 = np.asarray(Image.open(p1).convert('L').resize(s2), dtype=np.float32)
    return float(np.abs(a1 - a2).mean())


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    trn = sorted(p for p in (ROOT / 'train').iterdir() if p.is_file())
    HT = [dhash(p) for p in trn]
    say('=' * 96)
    say('leak/clean 划分一致性审计（dHash≤2 且 MAD≤20 两道闸）')
    say('=' * 96)
    say(f'  train 基准 {len(trn)} 张；判据 dHash≤{HAM_TH} 且 MAD≤{MAD_TH}')

    bad = 0
    for sp in ('test', 'val'):
        for sub in ('leak', 'clean'):
            names = sorted(p for p in (SUB / f'{sp}_{sub}' / 'images').iterdir() if p.is_file())
            n_cand = n_hit = 0
            hits = []
            for p in names:
                h = dhash(p)
                cs = [(HT[i], trn[i]) for i in range(len(trn))
                      if bin(h ^ HT[i]).count('1') <= HAM_TH]
                n_cand += 1 if cs else 0
                for _, tp in cs:
                    m = mad(p, tp)
                    if m <= MAD_TH:
                        n_hit += 1
                        hits.append((p.name, tp.name, m))
                        break
            say()
            say(f'--- {sp}_{sub}：{len(names)} 张 ---')
            say(f'  有 dHash≤2 候选的图：{n_cand} / {len(names)}')
            say(f'  其中**同时** MAD≤20（真近重复）：{n_hit}')
            for n, t, m in hits[:8]:
                say(f'      {n}  ≈  train/{t}   MAD={m:.2f}')
            if len(hits) > 8:
                say(f'      ...（共 {len(hits)} 张）')
            if sub == 'leak':
                if n_hit == len(names):
                    say('  ✓ leak 子集自洽：每张都能在 train 里找到真近重复。')
                else:
                    say(f'  ⚠ leak 子集有 {len(names) - n_hit} 张**找不到** MAD≤20 的近重复')
                    say('     ⇒ 要么 leak 名单来源与本次判据不一致，要么 MAD 口径变了，需查。')
                    bad += 1
            else:
                if n_hit == 0:
                    say('  ✓ clean 子集自洽：**没有**任何一张真近重复混进来。')
                    say('     ⇒ 分级剖面里 dHash≤2 的那些图全被 MAD 闸挡下（dHash 碰撞），')
                    say('       不是漏检 ⇒ P0 的泄漏量化输入无误。')
                else:
                    say(f'  ✗ clean 子集混进了 {n_hit} 张真近重复')
                    say('     ⇒ **P0 的 clean 读数被污染，泄漏量化必须重做**（make_leak_subsets.py 有 bug）。')
                    bad += 1

    say()
    say('=' * 96)
    say('审计结论')
    say('=' * 96)
    if bad == 0:
        say('  **划分自洽** ⇒ P0 的 leak/clean 输入可信，无需重做。')
    else:
        say('  **划分不自洽** ⇒ 先修 make_leak_subsets.py 并重跑 eval_leak_subsets.py，')
        say('     在此之前 P0 的所有子集读数作废。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
