# 架空输电线缺陷检测（YOLOv11s + 注意力机制失效分析）

用 **YOLOv11s** 在 **架空输电线路巡检图像**上做三类缺陷检测：
**break（断股）/ thunderbolt（雷击）/ wear（磨损）**，输入 896×896。

这个仓库的重点不是"跑了个 YOLO"，而是三件事：

1. **多模型正交对照** —— YOLOv5s / v11s / v12s × 注意力注入位置 × 优化器 × 分辨率，
   共 82 组训练 run 与 66 组独立评估 run，统一数据切分与评估口径。
2. **注意力模块为什么没用** —— 定量证明 CBAM 在该任务上近似**恒等映射**，
   并定位到根因：空间注意力输出退化为逐通道常数，被后继 BN 严格吸收。
3. **诚实的结果口径** —— 主动做跨 split 泄漏审计（test 中 48/179 张与 train 近重复），
   把指标明确限定为"同场景泛化"，不宣称新场景泛化能力。

---

## 结果

test 集（179 张 / 283 实例），3 个随机种子（42/43/44）均值。

| 模型 | 640 mAP@0.5 | 896 mAP@0.5 |
|---|---:|---:|
| YOLOv11s 基线 | 88.42 | 90.79 |
| + CBAM（BN 重标定） | 90.05 | 90.12 |
| + TripletAttention | 89.00 | **91.19** |
| YOLO26s 基线 | — | 91.35 |

（NMS IoU 0.7）

**最终对外口径**（NMS IoU 0.5，在 val 上独立选定，非 test 调参）：

**mAP@0.5 = 92.12%** ｜ mAP@0.5:0.95 = 54.23% ｜ P = 91.68% ｜ R = 85.17%

完整结果与 run 名对照见 [`results/main_results.md`](results/main_results.md)。

---

## 关键发现

**① 分辨率是唯一稳定的大杠杆。** 基线 640 → 896 提升 **+2.37 pt**；
任何注意力模块在 896 上的相对增益 ≤ **+0.40 pt**，且落在种子波动内。

**② CBAM 在这个任务上等价于恒等映射。** 排查链：

- 排除梯度问题（CBAM 梯度 6.2e-02，远高于 1e-3）
- 排除权重冻结（通道注意力每 epoch 位移 0.957%，与宿主层 0.975% 同量级）
- 定位到**输入有结构、输出没有**：层 2/4/6 的 z 存在大正 DC（+3.1~+4.9）→
  sigmoid 单边饱和成常数 1；层 8/9 的 z≈0 → 输出≈常数 0.5
- ⇒ 常数注意力图被下一层 BN 严格吸收（BN 对常数缩放在训练与推理下都不变，梯度恒 0）

**推论**：换成 SE / ECA 也救不回来——常数注意力图无论谁产出都会被 BN 吸收。

**③ 优化器的优劣会随分辨率反转。** 320 下 AdamW 优于 SGD（+0.016），
640 下反过来（−0.0095）。所以"哪个优化器更好"不能脱离分辨率谈。

**④ 指标有泄漏。** test 179 张中 48 张与 train 近似重复。
已另建 leak / clean 子集量化影响（见 `tools/check_leak_audit.py`、`tools/check_near_dup.py`）。

---

## 目录结构

```
├── src/
│   ├── triplet_att.py     TripletAttention 实现（WACV 2021，仅 300 参数）
│   ├── cbam.py            标准 CBAM（用于复现失效现象）
│   ├── cbam_bn.py         CBAM + 空间注意力读出后接 BN
│   ├── cbam_fixed.py      修复版：SA conv 加 bias + 空间中心化
│   ├── wiou_v3.py         WIoU v3 损失
│   ├── train_att.py       统一三臂训练入口（baseline / cbam_bn / triplet）
│   └── eval_test.py       评估入口
├── tools/
│   ├── check_params.py    参数量逐位核验（与理论增量对齐）
│   ├── check_leak_audit.py  跨 split 泄漏审计
│   ├── check_near_dup.py  近似重复检测
│   ├── viz_attention.py   注意力图可视化
│   └── benchmark_fps.py   端到端 FPS 基准
├── report/
│   ├── YOLOv11_CBAM_缺陷检测算法_v_final.pdf
│   ├── 模型改进与数据对照_v1.pdf
│   └── figures/yolo11_triplet_arch.{png,svg,drawio}
└── results/main_results.md
```

数据集、权重、训练产物均不入库（体积与版权原因）。

---

## 复现

```bash
pip install -r requirements.txt
```

数据：3 类 bbox 标注，`train 1428 / val 178 / test 179`（8:1:1，seed 42），
放成 `dataset/final_bbox/{images,labels}/{train,val,test}`，配置写在
`dataset/final_bbox/data.yaml`（`nc=3`，names 为 break / thunderbolt / wear）。

```bash
# 基线（噪声地板，3 个种子）
python src/train_att.py --att baseline --imgsz 896 --seed 42

# TripletAttention 臂
python src/train_att.py --att triplet --mode neck_sppf --imgsz 896 --seed 42

# CBAM + BN 重标定臂
python src/train_att.py --att cbam_bn --mode neck_sppf --imgsz 896 --seed 42

# 评估
python src/eval_test.py

# 参数量 / 泄漏 / 近重复核查
python tools/check_params.py
python tools/check_leak_audit.py
python tools/check_near_dup.py
```

脚本内所有路径都由 `Path(__file__)` 推导项目根，clone 到任意位置可直接跑。

---

## 训练配置

| 项 | 值 |
|---|---|
| epochs | 60 |
| batch | 8 |
| imgsz | 640 / 896 |
| optimizer | SGD（lr0=0.01、lrf=0.005、cos_lr） |
| AMP | 开 |
| workers | 0（Windows 下避免 spawn 重复导入主模块） |
| 各臂差异 | 仅分类头注入模块不同，其余超参逐项一致 |

---

## 说明

- 三类缺陷图像由现场巡检数据构建，SAM 辅助预标注后人工复核。
- 因版权与体积，数据集、权重、训练产物不上传。
- 本项目为算法侧实验，不含硬件部署部分。
