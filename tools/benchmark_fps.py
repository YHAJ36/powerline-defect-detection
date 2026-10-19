# -*- coding: utf-8 -*-
"""FPS 专门基准测试（论文用，替代日志里不可靠的 `Speed:` 行）。

口径
  - 端到端 = 读图 + letterbox/resize + 归一化 + 网络前向 + NMS
  - batch = 1，同一张测试图，imgsz 与训练一致（320 组用 320，640 组用 640）
  - GPU FP16（half），RTX 4060 Laptop，cudnn.benchmark 默认
  - **轮转（round-robin）测法**：所有模型轮流各测一轮，重复 4 轮，丢弃第 1 轮后取中位数
  - 每轮内：warmup 5 次 + 正式测 200 次

为什么必须轮转 + 重复 + 丢首轮（2026-10-10 实测，两次踩坑）
  踩坑一 冷启动：进程内第一个被测的模型命中一次性开销（CUDA 上下文 / cuDNN 句柄 / 显存池增长）。
      同一个 wire_v11_v3 @320：第 1 次 10.478 ms → 紧接着第 2 次 3.103 ms，差 3.4 倍。
      只跑一轮的话，"最先测"这个与模型无关的排序会把它系统性压低 3 倍以上。
  踩坑二 热漂移：8 秒烧机后 GPU 已 72 °C，同一架构 320 与 640 的裸前向竟测出同样的 14 ms
      （本该差 4 倍）——笔记本卡降频。逐模型连续测，越靠后的模型越热，顺序直接变成结论。
      → 改成轮转，热漂移对每个模型均摊；再丢弃第 1 轮、取剩余轮中位数。

为什么不用日志里的 Speed: 行
  那是训练中途单次 val 的读数，非稳态，且同架构同分辨率两次读数差异极大
  （640 neck 3.6 ms < 320 baseline 6.9 ms，物理上不可能）。

一条命令跑全部（脚本自己分派到两个解释器，v5 必须走 yolov5 隔离环境）：
    python code/tools/benchmark_fps.py

单项调试：
    python code/tools/benchmark_fps.py --spec wire_v11_v3:320
    {ROOT_S}/python code/tools/benchmark_fps.py \
        --engine yolov5 --spec wire_v5_640:640

输出：logs/fps_benchmark.txt（同时打印到 stdout）
"""
import argparse
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 tools/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


# 子进程的 stdout 会被父进程按 UTF-8 解码；Windows 默认 GBK，不统一就是乱码/编码错
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RUNS = ROOT_S + '/runs'
IMG = ROOT_S + '/dataset/final_bbox/images/test/firc_xianlan_101.jpg'
OUT = ROOT_S + '/logs/fps_benchmark.txt'

# 带 CBAM 的 best.pt 里 pickle 了 `cbam.CBAM` 引用，反序列化必须能找到该模块
# （v11/v12 两个 cbam.py 的类定义逐行一致，谁在前都能正确还原）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

YOLOV5_PY = ROOT_S + '/python'
YOLOV5_DIR = ROOT_S + '/code/baselines/yolov5'

WARMUP, MEASURE, ROUNDS = 5, 200, 4
BURN_IN_S = 5.0
KEEP_HOT = False        # True 时在每个计时轮前顶频（v5 路需要，见 --keep-hot）
KEEP_HOT_S = 2.0
CONF, IOU = 0.25, 0.45

# 用户指定的 6 组：name:imgsz
# 注：wire_v12s_v4 的训练分辨率是 320，此处按要求在 640 下推理（推理分辨率是运行时选择）；
#     v5 取 SGD 640 那组（总冠军 0.9070）；wire_v5_640_adamw 架构相同，FPS 相同。
DEFAULT_SPEC = [
    'wire_v11_v3:320',
    'wire_v11_640:640',
    'wire_v11_neck_v3:320',
    'wire_v11_640_neck:640',
    'wire_v12s_v4:640',
    'wire_v5_640:640',
]


def parse_spec(items):
    out = []
    for s in items:
        name, _, sz = s.rpartition(':')
        out.append((name, int(sz)))
    return out


def is_v5(name):
    return 'v5' in name


def gpu_probe():
    try:
        return subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return 'nvidia-smi 不可用'


class GpuWatch:
    """测量期间后台采样 GPU 状态。

    必须在**测量过程中**采样才有意义：每轮结束后再采样，采到的是空闲瞬间的低频。
    SM 频率直接决定算力，若两个模型不在同一频率下测，FPS 差多少就是假的多少。
    """

    def __init__(self, interval=0.15):
        self.samples = []          # (t, util%, sm_mhz, power_w, temp_c)
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                u, c, p, t = (x.strip() for x in out.split(','))
                self.samples.append((time.perf_counter(), int(u), int(c), float(p), int(t)))
            except Exception:
                pass
            self._stop.wait(0.15)

    def stop(self):
        self._stop.set()

    def during(self, t0, t1):
        v = [s for s in self.samples if t0 <= s[0] <= t1]
        if not v:
            return None
        return (int(statistics.median(s[2] for s in v)),      # SM MHz
                int(statistics.median(s[1] for s in v)),      # util %
                round(statistics.median(s[3] for s in v), 1),  # W
                int(statistics.median(s[4] for s in v)))       # ℃


def gpu_burn_in(seconds=BURN_IN_S):
    """把 GPU 从空闲低频状态拉到稳态；轮转测法下只需一次，不必烧到降频。"""
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return
    net = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1)).half().cuda().eval()
    a = torch.randn(8, 3, 640, 640, device='cuda', dtype=torch.half)
    t0 = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t0 < seconds:
            for _ in range(10):
                net(a)
            torch.cuda.synchronize()
    print(f'[烧机] {seconds:.0f}s 完成    GPU: {gpu_probe()}', flush=True)


# ---------------------------------------------------------------- ultralytics
def load_ultralytics(name):
    from ultralytics import YOLO
    return YOLO(f'{RUNS}/{name}/weights/best.pt')


def measure_ultralytics(model, imgsz, rounds=ROUNDS):
    import torch

    kw = dict(imgsz=imgsz, batch=1, device=0, half=True, conf=CONF, iou=IOU, verbose=False)
    model.predict(source=[IMG] * WARMUP, **kw)
    torch.cuda.synchronize()

    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        res = model.predict(source=[IMG] * MEASURE, **kw)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / MEASURE * 1000.0)
        parts = {k: sum(float(r.speed[k]) for r in res) / len(res)
                 for k in ('preprocess', 'inference', 'postprocess')}
    return out, parts


# -------------------------------------------------------------------- yolov5
def load_yolov5(name):
    import torch

    sys.path.insert(0, YOLOV5_DIR)
    from models.experimental import attempt_load

    dev = torch.device('cuda:0')
    m = attempt_load(f'{RUNS}/{name}/weights/best.pt', device=dev)
    return m.half().eval()


def measure_yolov5(model, imgsz, rounds=ROUNDS):
    import cv2
    import numpy as np
    import torch

    from utils.augmentations import letterbox
    from utils.general import non_max_suppression

    dev = torch.device('cuda:0')
    stride = int(model.stride.max())

    def infer():
        """复刻 detect.py 的单帧端到端链路（含读图，与 ultralytics 侧对齐）。"""
        img0 = cv2.imread(IMG)
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)                  # BGR→RGB, HWC→CHW
        t = torch.from_numpy(np.ascontiguousarray(img)).to(dev).half() / 255.0
        with torch.no_grad():
            pred = model(t.unsqueeze(0))[0]
        return non_max_suppression(pred, CONF, IOU, None, False, max_det=300)

    for _ in range(WARMUP):
        infer()
    torch.cuda.synchronize()

    out = []
    for _ in range(rounds):
        if KEEP_HOT:
            # YOLOv5 这条路 CPU 占比高、GPU 占空比低，驱动会把频率降到 ~1000 MHz，
            # 而 ultralytics 那条路稳定 2490 MHz。不顶频就是拿两个频率下的数比 FPS。
            gpu_burn_in(KEEP_HOT_S)
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            infer()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / MEASURE * 1000.0)

    # 分段耗时（单独一趟，内部同步扰动绝对值，仅供拆分参考）
    n, acc = 50, [0.0, 0.0, 0.0]
    for _ in range(n):
        a = time.perf_counter()
        img0 = cv2.imread(IMG)
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(img)).to(dev).half() / 255.0
        b = time.perf_counter()
        with torch.no_grad():
            pred = model(t.unsqueeze(0))[0]
        torch.cuda.synchronize()
        c = time.perf_counter()
        non_max_suppression(pred, CONF, IOU, None, False, max_det=300)
        torch.cuda.synchronize()
        d = time.perf_counter()
        acc[0] += (b - a) * 1000
        acc[1] += (c - b) * 1000
        acc[2] += (d - c) * 1000
    parts = dict(zip(('preprocess', 'inference', 'postprocess'), [v / n for v in acc]))
    return out, parts


def run_round_robin(rows, engine, watch=None):
    """轮转：所有模型轮流各测一轮，重复 ROUNDS 轮。返回 {key: (rounds, parts, clocks)}。"""
    load = load_yolov5 if engine == 'yolov5' else load_ultralytics
    measure = measure_yolov5 if engine == 'yolov5' else measure_ultralytics

    models = {}
    for key in rows:
        if not Path(f'{RUNS}/{key[0]}/weights/best.pt').exists():
            print(f'!! 缺权重 {RUNS}/{key[0]}/weights/best.pt', flush=True)
            continue
        models[key] = load(key[0])
    print(f'已载入 {len(models)} 个模型，开始轮转 {ROUNDS} 轮', flush=True)

    acc = {k: [] for k in models}
    parts, clocks = {}, {}
    for r in range(ROUNDS):
        for key, model in models.items():
            t0 = time.perf_counter()
            ms, parts[key] = measure(model, key[1], rounds=1)
            t1 = time.perf_counter()
            acc[key].append(ms[0])
            if watch is not None:
                clocks.setdefault(key, []).append(watch.during(t0, t1))
        print(f'  轮 {r + 1}/{ROUNDS}  {gpu_probe()}', flush=True)
    return {k: (acc[k], parts[k], clocks.get(k)) for k in models}


def emit(results, rows):
    """子进程只输出机器可解析行，父进程收集后统一排版。"""
    for key in rows:
        if key not in results:
            continue
        rounds, p, clocks = results[key]
        med = statistics.median(rounds[1:])          # 丢弃第 1 轮（各模型的冷启动）
        clk = [c[0] for c in clocks] if clocks else []
        print(f'RESULT|{key[0]}|{key[1]}|{med:.4f}|{1000.0 / med:.2f}|'
              f'{" ".join(f"{v:.2f}" for v in rounds)}|'
              f'{p["preprocess"]:.3f}/{p["inference"]:.3f}/{p["postprocess"]:.3f}|'
              f'{" ".join(str(v) for v in clk)}', flush=True)


def table(results, rows):
    L = ['=' * 100,
         'FPS 基准（端到端：读图 + letterbox + 归一化 + 前向 + NMS）',
         f'RTX 4060 Laptop / FP16 / batch=1 / 每轮 warmup {WARMUP} + 测 {MEASURE}；'
         f'轮转 {ROUNDS} 轮，丢弃第 1 轮后取中位数',
         f'测试图 {IMG}',
         '=' * 100,
         f'{"run":<22}{"imgsz":>6}{"ms/张":>10}{"FPS":>8}   '
         f'{"各轮原始值(第1轮已丢弃)":<28}{"分段(前处理/前向/NMS)":<24}{"测量中 SM 频率(MHz)":<20}',
         '-' * 118]
    for key in rows:
        if key not in results:
            continue
        rounds, p, clocks = results[key]
        med = statistics.median(rounds[1:])
        rstr = '[' + ' '.join(f'{v:.2f}' for v in rounds) + ']'
        pstr = f'{p["preprocess"]:.3f} / {p["inference"]:.3f} / {p["postprocess"]:.3f}'
        clk = ' '.join(str(c[0]) for c in clocks) if clocks else '-'
        L.append(f'{key[0]:<22}{key[1]:>6}{med:>10.3f}{1000.0 / med:>8.1f}   '
                 f'{rstr:<28}{pstr:<24}{clk:<20}')
    L += ['=' * 118,
          '注 1：轮转测法——所有模型轮流各测一轮再进下一轮，使笔记本 GPU 的热漂移对各模型均摊。',
          '注 2：第 1 轮含各模型自身的冷启动一次性开销，一律丢弃，不参与中位数。',
          '注 3：SM 频率为测量窗口内 nvidia-smi 的中位数；频率不一致的组之间不可直接比 FPS。',
          '注 4：v5 无框架自报分段，其分段为单独一趟插桩测量，绝对值受同步扰动，仅作占比参考。',
          '注 5：v5 在独立解释器进程中测量（隔离环境），跨进程热状态与 v11/v12 不完全一致。']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['auto', 'ultralytics', 'yolov5'], default='auto')
    ap.add_argument('--spec', nargs='+', default=DEFAULT_SPEC,
                    help='name:imgsz ...（默认用户指定的 6 组）')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--no-burn', action='store_true', help='跳过烧机（不推荐）')
    ap.add_argument('--keep-hot', action='store_true',
                    help='每个计时轮前烧机顶频（YOLOv5 路 GPU 占空比低会掉频，必须开）')
    args = ap.parse_args()

    global KEEP_HOT
    KEEP_HOT = args.keep_hot

    rows = parse_spec(args.spec)
    watch = GpuWatch()
    watch.start()

    if args.engine == 'yolov5':
        if not args.no_burn:
            gpu_burn_in()
        emit(run_round_robin(rows, 'yolov5', watch), rows)
        watch.stop()
        return

    if args.engine == 'auto':
        mine = [r for r in rows if not is_v5(r[0])]
        theirs = [r for r in rows if is_v5(r[0])]
    else:
        mine, theirs = rows, []

    if mine:
        if not args.no_burn:
            gpu_burn_in()
        results = run_round_robin(mine, 'ultralytics', watch)
    else:
        results = {}

    if theirs:
        if not Path(YOLOV5_PY).exists():
            print(f'!! 缺 yolov5 解释器 {YOLOV5_PY}，跳过 {[n for n, _ in theirs]}', flush=True)
        else:
            spec = [f'{n}:{s}' for n, s in theirs]
            print(f'>>> 分派到 yolov5 环境：{" ".join(spec)}', flush=True)
            cp = subprocess.run(
                [YOLOV5_PY, str(Path(__file__).resolve()), '--engine', 'yolov5',
                 '--spec', *spec] + (['--keep-hot'] if KEEP_HOT else []),
                cwd=YOLOV5_DIR, capture_output=True, text=True,
                encoding='utf-8', errors='replace')
            for ln in cp.stdout.splitlines():
                if ln.startswith('RESULT|'):
                    parts_ = ln.split('|')
                    n, sz, med, _fps, rs, ps, clk = parts_[1:8]
                    p = dict(zip(('preprocess', 'inference', 'postprocess'),
                                 (float(v) for v in ps.split('/'))))
                    clocks = [(int(v), None, None, None) for v in clk.split()] or None
                    results[(n, int(sz))] = ([float(v) for v in rs.split()], p, clocks)
                else:
                    safe = ln.encode('gbk', 'replace').decode('gbk')
                    print('   [v5]', safe, flush=True)
            if cp.returncode != 0:
                print(f'!! yolov5 侧退出码 {cp.returncode}', flush=True)
                print(cp.stderr[-1500:], flush=True)
    watch.stop()

    txt = table(results, rows)
    print('\n' + txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt + '\n', encoding='utf-8')
    print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
