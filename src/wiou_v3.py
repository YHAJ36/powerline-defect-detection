# -*- coding: utf-8 -*-
"""WIoU v3（Wise-IoU v3）边界框回归损失，供 Phase 1 E1 使用。

来源：https://github.com/Instinct323/Wise-IoU  branch v2  iou.py
      （对应论文 Tong et al., "Wise-IoU: Bounding Box Regression Loss with
       Dynamic Focusing Mechanism", arXiv:2301.10051）

本文件只做两件事：
1. 把官方 `IouLoss(ltype='WIoU', monotonous=False)` 的公式原样搬到
   Ultralytics 的 xyxy 张量上（float32 计算，保持数值稳定）。
2. 提供 `WIoUv3BboxLoss`，替换 `ultralytics.utils.loss.BboxLoss` 中的
   CIoU 项；DFL 项与官方 Ultralytics 实现完全一致。

官方默认超参（**E1 使用这一组**）：
    momentum=0.01, alpha=1.7, delta=2.7
教程/部分复现常用 alpha=1.9, delta=3.0；按用户要求，只有 E1 为正收益时
才补跑一组 1.9/3.0 做对照。

注意：`iou_mean` 是 criterion 内部的 buffer，不在 model.state_dict 里。
若训练被中断后用 `resume=True` 续训，`iou_mean` 会从 1.0 重新开始，
前几个 epoch 的聚焦权重会有一个短暂过渡；正式结果不依赖续训。
"""
import torch
import torch.nn as nn

from ultralytics.utils.loss import DFLoss, bbox2dist


class WIoUv3Loss(nn.Module):
    """WIoU v3 回归损失（输入/输出均为 xyxy，返回逐框 loss，shape (N,1)）。

    官方实现里变量 `iou` 实际是 **IoU loss**（1 - IoU），
    `iou_mean` 跟踪的也是这个 loss 的滑动平均，不是 IoU 本身。
    这里保留同样的命名，避免与论文公式对不上。
    """

    def __init__(self, momentum: float = 1e-2, alpha: float = 1.7, delta: float = 2.7):
        super().__init__()
        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 与 ultralytics 的 bbox_iou 一样用 float32，AMP 下更稳
        pred = pred.float()
        target = target.float()
        eps = 1e-7

        pred_xy = (pred[..., :2] + pred[..., 2:4]) / 2
        pred_wh = pred[..., 2:4] - pred[..., :2]
        target_xy = (target[..., :2] + target[..., 2:4]) / 2
        target_wh = target[..., 2:4] - target[..., :2]

        min_coord = torch.minimum(pred[..., :4], target[..., :4])
        max_coord = torch.maximum(pred[..., :4], target[..., :4])

        wh_inter = torch.relu(min_coord[..., 2:4] - max_coord[..., :2])
        s_inter = torch.prod(wh_inter, dim=-1)
        s_union = torch.prod(pred_wh, dim=-1) + torch.prod(target_wh, dim=-1) - s_inter
        s_union = s_union.clamp_min(eps)

        wh_box = max_coord[..., 2:4] - min_coord[..., :2]
        l2_box = torch.square(wh_box).sum(dim=-1).clamp_min(eps)
        l2_center = torch.square(pred_xy - target_xy).sum(dim=-1)

        iou_loss = 1 - s_inter / s_union

        if self.training:
            with torch.no_grad():
                self.iou_mean.mul_(1 - self.momentum)
                self.iou_mean.add_(self.momentum * iou_loss.detach().mean())

        # WIoU v1：距离注意力用 detach 的 l2_box，不回传
        dist = torch.exp(l2_center / l2_box.detach())
        loss = dist * iou_loss

        # WIoU v3：动态非单调聚焦
        beta = iou_loss.detach() / self.iou_mean.clamp_min(eps)
        divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
        loss = loss * beta / divisor

        return loss.unsqueeze(-1)  # (N,1)，与 CIoU 的返回形状保持一致


class WIoUv3BboxLoss(nn.Module):
    """替换 ultralytics.utils.loss.BboxLoss：CIoU -> WIoU v3，DFL 不变。"""

    def __init__(self, reg_max: int = 16, alpha: float = 1.7, delta: float = 2.7,
                 momentum: float = 1e-2):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.wiou = WIoUv3Loss(momentum=momentum, alpha=alpha, delta=delta)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if fg_mask.any():
            loss_iou = self.wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = (loss_iou * weight).sum() / target_scores_sum
        else:
            loss_iou = pred_bboxes.sum() * 0.0

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


# patch 的默认超参。必须是模块级变量：torch.save 会 pickle criterion 的类型，
# 若把 PatchedBboxLoss 定义在函数内部，保存 checkpoint 时会报
# "Can't get local object 'patch_ultralytics.<locals>...'"。
ALPHA = 1.7
DELTA = 2.7
MOMENTUM = 1e-2


class PatchedBboxLoss(WIoUv3BboxLoss):
    """模块级别名类，便于 torch.save/load 按引用 pickle。"""

    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max, alpha=ALPHA, delta=DELTA, momentum=MOMENTUM)


def patch_ultralytics(alpha: float = 1.7, delta: float = 2.7, momentum: float = 1e-2):
    """把 ultralytics 的 BboxLoss 全局替换为 WIoU v3 版本。

    必须在 `YOLO(...).train()` / `trainer.get_model()` 触发 criterion 初始化之前调用。
    `v8DetectionLoss.__init__` 在运行时查模块全局 `BboxLoss`，因此直接赋模块属性即可。
    """
    global ALPHA, DELTA, MOMENTUM
    ALPHA, DELTA, MOMENTUM = float(alpha), float(delta), float(momentum)

    import ultralytics.utils.loss as loss_module
    loss_module.BboxLoss = PatchedBboxLoss
    return PatchedBboxLoss
