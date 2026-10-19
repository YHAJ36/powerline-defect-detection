# -*- coding: utf-8 -*-
"""640 注意力臂统一训练脚本：噪声地板（baseline×3 种子）+ CBAM+BN + Triplet。

实验动机
    报告核心算法要求是"优化算法 = CBAM"，但报告第五章定量证明 CBAM 近似恒等映射，
    根因定位到 SA 读出头（DC 饱和 + 幅度衰减）。第一版修复 cbam_fixed 已失败。
    本脚本跑三件事：
      ① **噪声地板** —— 640 baseline 跑 3 个种子（42/43/44）。test 集仅 179 图 283 实例，
         评估口径下 CBAM 消融的效应只有 ±0.013（mAP50），落在噪声内。没有 3 种子的
         mAP 极差，任何"涨了/跌了"都不可判读。
      ② **CBAM+BN** —— 只给 SA 读出头加 BN（cbam_bn.py）。
      ③ **Triplet** —— triplet_att.TripletAttention，其 AttentionGate 天然含"BN→sigmoid"，
         与 ② 同机理、不同实现，用来互为印证。

口径
    超参与基线 run `wire_v11_640` 逐项一致，
    唯一差别是本脚本的 project 指向 `runs`。
    A/B 两臂的 args.yaml 除模块实现外应逐项一致。

    ⚠ **workers 的口径坑（2026-10-15 实测）**：
      基线 run 的 args.yaml 记的是 `workers: 4`，而它之后所有 640 臂
      （neck/backbone/both/cbamfix）以及 896 基线都是 `workers: 0`。
      实测速度差异很大：workers=4 是 **25 s/epoch（60 轮 25 min）**，
      workers=0 是 **70 s/epoch（60 轮 70 min）** —— 差 2.8 倍（dataloading 不重叠）。
      本脚本取 **workers=0**，理由：被正式实验那批 CBAM 臂（基线配置的
      neck/backbone/both）就是 workers=0，"CBAM+BN vs 原版 CBAM"的对比必须同管线；
      且噪声地板的 3 个种子全部用本脚本重跑（不引用归档的 0.8992，那是 workers=4），
      以免"workers/管线不同"混进噪声地板、污染判读。归档 0.8992 只作参照记录。

注入方式（参照 CBAM 的失效分析结论）
    必须在 `DetectionTrainer.get_model()` **重建之后**注入；在 train() 之前注入会被
    train() 内部的 yaml 重建静默丢弃。故此处子类化 get_model，并打印参数量供核验。

用法
    python {ROOT_S}/src/train_att_640.py --att baseline --seed 43
    ... --att cbam_bn --seed 42
    ... --att triplet --seed 42 --imgsz 896
续训（被中断后接上，不从头）：
    ... --att cbam_bn --seed 43 --resume --name att_v11_640_cbam_bn_s43
冒烟（不改正式产物）：
    ... --att cbam_bn --epochs 1 --fraction 0.03 --name smoke_cbambn

⚠ 文件名里的 640 是**历史命名**（第一批 640 用），实际分辨率由 `--imgsz` 决定（默认 640）。
⚠ `--resume` 的口径坑（2026-10-15 实测崩溃后查明，详见 make_trainer 的注释）：
   ultralytics 续训是**按 baseline yaml 重建模型**再载 ckpt 权重（键名取交集），
   **不是**从 ckpt 里 pickle 的模型接上。故：
     · 必须同时传 `trainer=`（本脚本的 resume 分支已如此），否则优化器 param group
       对不上直接 `ValueError` 崩；即便不崩，注入层也会退化成随机初值；
     · `get_model()` **会**被调用（经 `setup_model`），但**注入参数量打印走的是另一条路径**，
       且 `resume_training` 不搬 model 权重 —— 续训后仍必须核 `best.pt` 参数量，
       用 `code/tools/check_run_params.py`（这是唯一的机器判据）。
"""
import argparse
import sys
from pathlib import Path

from pathlib import Path

# 项目根目录：本文件位于 src/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

if ROOT_S + '/src' not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))      # cbam_bn / triplet_att / cbam

EXP = ROOT_S + '/runs'                    # 训练产物目录
DATA = ROOT_S + '/dataset/final_bbox/data.yaml'
WEIGHTS = 'D:/1/models/yolo11s.pt'

# 与基线配置的 model.train(...) **逐项一致**
# imgsz 由 --imgsz 覆盖（默认 640）；其余各项与基线配置逐项一致。
BASE = dict(
    data=DATA, imgsz=640, batch=8,
    optimizer='SGD', lr0=0.01, lrf=0.005,
    momentum=0.937, weight_decay=0.0005,
    cos_lr=True, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.8, mixup=0.1, copy_paste=0.1,
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
    close_mosaic=5, label_smoothing=0.1, dropout=0.0,
    amp=True, workers=0, device=0,
    project=EXP, exist_ok=True,
)

BACKBONE = [2, 4, 6, 8]      # C3k2
SPPF = 9                     # SPPF
MODES = {'neck_sppf': [SPPF], 'backbone': BACKBONE, 'both': BACKBONE + [SPPF]}


def get_channels(m):
    """模块输出通道数。"""
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)   # ultralytics Conv 包装了 nn.Conv2d
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def make_attention(att):
    """返回 (ctor, 标签)。ctor(ch) -> nn.Module。"""
    if att == 'cbam_bn':
        from cbam_bn import CBAM
        return (lambda ch: CBAM(ch)), 'CBAM+BN'
    if att == 'triplet':
        from triplet_att import TripletAttention
        return (lambda ch: TripletAttention()), 'Triplet'
    raise ValueError(f'未知注意力类型 {att}')


def inject(model, att, mode):
    """把注意力模块包到目标层后面；返回注入的层索引。"""
    seq = model.model
    idx = MODES[mode]
    ctor, label = make_attention(att)
    for i in idx:
        ch = get_channels(seq[i])
        attn = ctor(ch)
        w = nn.Sequential(seq[i], attn)
        # 保留 forward 路由与自省属性（漏掉 .f 会让 reshape 路由断掉）
        w.f = seq[i].f
        w.i = seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w
    n = sum(p.numel() for p in model.parameters())
    print(f'>>> 注入 {label}  mode={mode}  layers={idx}；全模型参数量 {n:,}')
    return idx


def make_trainer(att, mode):
    """按 att/mode 注入的 DetectionTrainer 子类（在 get_model 重建后注入）。

    ⚠ 续训（resume）为什么必须也传这个 trainer（2026-10-15 实测崩溃后查明）
        ultralytics 续训**不是**从 ckpt 里 pickle 的模型接上，而是
        `setup_model()`（engine/trainer.py:618-636）：
            weights, ckpt = attempt_load_one_weight(last.pt)   # weights = ckpt 的 ema（已训、含注入层）
            cfg = weights.yaml                                 # ← 这是 **baseline yaml**
            self.model = self.get_model(cfg=cfg, weights=weights)
        而 `DetectionModel.load()` 是按键名**取交集**载入 ⇒ 注入层的键在 baseline 模型里
        不存在、被静默丢掉（日志 `Transferred 487/499` 即此）。后果两条：
          ① 优化器状态 dict 是注入模型的，模型却是基线 ⇒ param group 数量/大小不符 ⇒
             `ValueError: loaded state dict contains a parameter group that doesn't match` 直接崩；
          ② 即使不崩，inject() 新造的注入层是**随机初值**（`resume_training` 只搬
             optimizer / ema / updates，**不搬 model 权重**），而 best.pt 派生自 EMA
             ⇒ 臂被静默污染，只看 mAP 分不出。
        所以这里做两件事：**续训也传本 trainer**（保证架构一致、优化器能对上），
        并在注入后把 ckpt 里的注入层权重**搬回来**（保证不是随机初值）。
    """

    class AttDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if att != 'baseline':
                idx = inject(model, att, mode)
                # 续训：若传入的 weights（ckpt 的 ema）里确实带着注入层权重，就整份搬回来
                if isinstance(weights, nn.Module):
                    sd = weights.float().state_dict()
                    keys = [k for k in model.state_dict()
                            if any(k.startswith(f'model.{i}.1.') for i in idx)]
                    if keys and all(k in sd for k in keys):
                        model.load_state_dict(
                            {k: v for k, v in sd.items() if k in model.state_dict()},
                            strict=False)
                        print(f'>>> 续训：已从 ckpt 恢复注入层权重 {len(keys)} 个张量'
                              f'（否则这些层会是随机初值）')
                    else:
                        print(f'>>> 新训：注入层 {len(keys)} 个张量为初值'
                              f'（ckpt 里无注入层权重）')
            return model

    return AttDetectionTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--att', choices=['baseline', 'cbam_bn', 'triplet'], required=True)
    p.add_argument('--mode', choices=sorted(MODES), default='neck_sppf',
                   help='注入位置；默认 neck_sppf（报告表 10 唯一非负方向）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--name', default=None)
    p.add_argument('--save-period', type=int, default=20)
    p.add_argument('--resume', action='store_true',
                   help='从该 run 的 last.pt 续训（中断后接上，不从头重跑）')
    # 冒烟覆盖（正式跑批一律不传）
    p.add_argument('--fraction', type=float, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--project', default=None)
    return p.parse_args()


def main():
    args = parse_args()

    kw = dict(BASE)
    kw['epochs'] = args.epochs
    kw['seed'] = args.seed
    kw['imgsz'] = args.imgsz
    kw['save_period'] = args.save_period
    if args.fraction is not None:
        kw['fraction'] = args.fraction
    if args.device is not None:
        kw['device'] = args.device
    if args.project is not None:
        kw['project'] = args.project
    prefix = (f'nf_v11_{args.imgsz}' if args.att == 'baseline'
              else f'att_v11_{args.imgsz}_{args.att}')
    kw['name'] = args.name or f'{prefix}_s{args.seed}'

    print(f'=== att={args.att}  mode={args.mode}  seed={args.seed}  '
          f'imgsz={args.imgsz}  epochs={args.epochs}  name={kw["name"]} ===')
    if args.fraction is not None:
        print(f'    ⚠ 冒烟：fraction={args.fraction}（非正式口径）')

    if args.resume:
        # 续训：从该 run 的 last.pt 接上。ultralytics 的 resume 会从 ckpt 里读回
        # 原始 train_args（含 save_dir / epochs），故此处不传 kw，只给 resume。
        # ⚠ 必须同时传 trainer=本批 trainer，否则模型按 baseline yaml 重建 ⇒
        #   优化器 param group 对不上直接崩（且注入层会变随机初值）。见 make_trainer 注释。
        last = Path(kw['project']) / kw['name'] / 'weights' / 'last.pt'
        if not last.exists():
            raise SystemExit(f'❌ 无法续训：{last} 不存在')
        if not (Path(kw['project']) / kw['name'] / 'weights' / 'best.pt').exists():
            print(f'⚠ {kw["name"]} 尚无 best.pt —— 崩得太早，续训可继续，但该 run 目前只有 last.pt')
        print(f'>>> 续训：{last}')
        YOLO(str(last)).train(trainer=make_trainer(args.att, args.mode), resume=True)
        return

    model = YOLO(WEIGHTS)
    model.train(trainer=make_trainer(args.att, args.mode), **kw)


if __name__ == '__main__':
    main()
