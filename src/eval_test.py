# -*- coding: utf-8 -*-
"""在 test 集上评估 640 版 5 组 ultralytics 模型（v11×3 + v12×2）的 best.pt。

口径同 eval_test_sgd.py / eval_test_adamw.py，仅 run 名换 640 版、imgsz 改 640。
v11 baseline（wire_v11_640）此前已单独评过（eval_test_640.log），此处不重复。

Windows spawn 下必须 workers=0 + __main__ 守卫。
CBAM 模型的 best.pt 反序列化需要 `cbam` 可导入，故把 yolov11 目录加入 sys.path
（yolov11/cbam.py 与 yolov12/cbam.py 内容相同，类名都是 CBAM）。
"""
import os
import sys

from pathlib import Path

# 项目根目录：本文件位于 src/，向上 1 级即项目根
ROOT = Path(__file__).resolve().parents[1]
ROOT_S = ROOT.as_posix()          # 便于直接拼字符串


for d in (ROOT_S + '/src', ROOT_S + '/src'):
    if d not in sys.path:
        sys.path.insert(0, d)

from ultralytics import YOLO

NAMES = [
    'wire_v11_640_neck',
    'wire_v11_640_backbone',
    'wire_v11_640_both',
    'wire_v12_640',
    'wire_v12_640_cbam',
]


def main():
    print('run                              mAP50    mAP50-95   P        R', flush=True)
    for n in NAMES:
        m = YOLO(f'{ROOT_S}/runs/{n}/weights/best.pt')
        res = m.val(
            data=ROOT_S + '/dataset/final_bbox/data.yaml',
            split='test', imgsz=640, batch=8, workers=0, device=0,
            project=ROOT_S + '/runs', name=f'{n}_test', exist_ok=True, verbose=True,
        )
        b = res.box
        print(f'{n:<32} {b.map50:.4f}   {b.map:.4f}   {b.mp:.4f}  {b.mr:.4f}', flush=True)


if __name__ == '__main__':
    main()
