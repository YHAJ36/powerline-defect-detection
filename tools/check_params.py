# -*- coding: utf-8 -*-
"""参数量补测：v12s baseline / v12s+CBAM / v5s（build + count，秒级，不训练）。

背景：640 四套总表需要"参数量"列。v11 体系已有（check_cbam_params.py，9,428,953 基准），
      缺 v12s、v12s+CBAM、v5s 三个数。

一条命令跑全部（脚本自己分派到两个解释器，v5 必须走 yolov5 隔离环境）：
    python code/tools/check_params_all.py

两种取数方式都给，互为印证：
  - 理论核验：加载起点权重 / build 官方 yaml，数参数量（nc=80 或 nc=3）
  - 产物核验：读训练产物 best.pt（nc=3，与数据集一致）—— 这是报告里用的口径

输出：logs/params_all.txt
"""
import argparse
import subprocess
import sys
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 tools/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


RUNS = ROOT_S + '/runs'
V11_DIR = ROOT_S + '/src'
V12_DIR = ROOT_S + '/src'
V11_BASE = 'D:/1/models/yolo11s.pt'
V12_BASE = 'D:/1/models/yolo12s.pt'
OUT = ROOT_S + '/logs/params_all.txt'

YOLOV5_PY = ROOT_S + '/python'
YOLOV5_DIR = ROOT_S + '/code/baselines/yolov5'

# 产物核验：报告四套总表用到的全部 v12 / v5 组
V12_RUNS = ['wire_v12s_v4', 'wire_v12s_cbam_v4', 'wire_v12_640', 'wire_v12_640_cbam']
V5_RUNS = ['wire_v5_v2', 'wire_v5_640', 'wire_v5_adamw', 'wire_v5_640_adamw']


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def ultralytics_part():
    sys.path.insert(0, V11_DIR)
    sys.path.insert(0, V12_DIR)
    from ultralytics import YOLO
    from cbam import inject_cbam  # 两个目录同名模块；v11/v12 的 inject_cbam 接口一致

    lines = ['-' * 76]
    lines.append('【理论核验】build 起点权重 + 注入（不训练）')
    lines.append('-' * 76)
    lines.append(f'{"模型":<28}{"params":>14}{"delta":>12}   注入层')

    # v11 各 mode 已有专用脚本 check_cbam_params.py，这里只补 v12，v11 仅列基线便于对照
    base11 = n_params(YOLO(V11_BASE).model)
    lines.append(f'{"yolo11s.pt (nc=80) 基线":<28}{base11:>14,}{0:>+12,}   -')
    base12 = n_params(YOLO(V12_BASE).model)
    lines.append(f'{"yolo12s.pt (nc=80) 基线":<28}{base12:>14,}{0:>+12,}   -')
    m12 = YOLO(V12_BASE)
    idx = inject_cbam(m12, {'A2C2f'}, mode='all')
    p12 = n_params(m12.model)
    lines.append(f'{"yolo12s.pt + CBAM(A2C2f)":<28}{p12:>14,}{p12 - base12:>+12,}   '
                 f'{idx} 共 {len(idx)} 个')

    lines.append('')
    lines.append('-' * 76)
    lines.append('【产物核验】训练产物 best.pt（nc=3，与数据集一致）—— 报告采用此口径')
    lines.append('-' * 76)
    lines.append(f'{"run":<30}{"params":>14}{"delta":>12}')
    b12 = None
    for n in V12_RUNS:
        w = f'{RUNS}/{n}/weights/best.pt'
        if not Path(w).exists():
            lines.append(f'{n:<30}{"缺权重":>14}')
            continue
        p = n_params(YOLO(w).model)
        if b12 is None:
            b12 = p
        lines.append(f'{n:<30}{p:>14,}{p - b12:>+12,}')
    return lines


def yolov5_part():
    import contextlib
    import io

    import torch
    sys.path.insert(0, YOLOV5_DIR)
    from models.experimental import attempt_load
    from models.yolo import Model
    from utils.general import LOGGER

    # yolov5 的 LOGGER 在模块导入时就绑定了真实 stdout，redirect_stdout 拦不住，直接压级别
    LOGGER.setLevel(50)

    lines = []
    lines.append('-' * 76)
    lines.append('【v5s】YOLOv5 独立环境')
    lines.append('-' * 76)
    lines.append(f'{"模型":<30}{"params":>14}{"delta":>12}')

    with contextlib.redirect_stdout(io.StringIO()):   # 屏蔽建图时打印的逐层摘要
        built = Model('models/yolov5s.yaml', ch=3, nc=3)
    pb = n_params(built)
    lines.append(f'{"build yolov5s.yaml (nc=3)":<30}{pb:>14,}{0:>+12,}')

    base = None
    for n in V5_RUNS:
        w = f'{RUNS}/{n}/weights/best.pt'
        if not Path(w).exists():
            lines.append(f'{n:<30}{"缺权重":>14}')
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            em = attempt_load(w, device='cpu', fuse=False)  # 不融合 BN，与 v11 的读法同口径
        p = n_params(em)
        if base is None:
            base = p
        lines.append(f'{n:<30}{p:>14,}{p - base:>+12,}')
    lines.append('（v5 各组架构/类别数相同，参数量理应完全一致）')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['auto', 'ultralytics', 'yolov5'], default='auto')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    if args.engine == 'yolov5':
        print('\n'.join(yolov5_part()))
        return

    out = ['=' * 76, '参数量补测：v12s / v12s+CBAM / v5s', '=' * 76]
    out += ultralytics_part()

    if Path(YOLOV5_PY).exists():
        print('>>> 分派到 yolov5 环境数 v5s 参数量', flush=True)
        cp = subprocess.run([YOLOV5_PY, str(Path(__file__).resolve()), '--engine', 'yolov5'],
                            cwd=YOLOV5_DIR, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
        if cp.returncode == 0:
            out += cp.stdout.rstrip('\n').splitlines()
        else:
            out.append(f'!! yolov5 侧退出码 {cp.returncode}')
            out.append(cp.stderr[-1500:])
    else:
        out.append(f'!! 缺 yolov5 解释器 {YOLOV5_PY}，跳过 v5s')

    # ultralytics 每次运行都会打印 settings 重置警告（本机已知问题），报告产物里不需要
    NOISE = ('Ultralytics settings', 'View Ultralytics Settings', 'Update Settings with')
    out = [l for l in out if not any(n in l for n in NOISE)]
    txt = '\n'.join(out)
    print('\n' + txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt + '\n', encoding='utf-8')
    print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
