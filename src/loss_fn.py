import torch
import torch.nn.functional as F

from eval.metrics import circle_iou_torch

def circle_loss(
    preds,
    targets,
    w_center=1.0,
    w_radius=2.0,
    w_iou=1.0,
    w_iou_c1=1.0,
    w_iou_c2=1.0,
    beta=0.02,
    iou_mode="one_minus",
    eps=1e-7,
):
    """
    preds/targets: (B, 6) = [cx1, cy1, r1, cx2, cy2, r2], normalized to [0,1]
    """

    preds_v = preds.view(-1, 2, 3)
    targets_v = targets.view(-1, 2, 3)

    pc, pr = preds_v[..., :2], preds_v[..., 2]
    tc, tr = targets_v[..., :2], targets_v[..., 2]

    lc = F.smooth_l1_loss(pc, tc, beta=beta, reduction="mean")
    lr = F.smooth_l1_loss(pr, tr, beta=beta, reduction="mean")

    base = w_center * lc + w_radius * lr

    if w_iou == 0.0:
        return base

    iou = circle_iou_torch(preds, targets)

    if iou_mode == "log":
        per_circle = -torch.log(iou.clamp_min(eps))
    else:
        per_circle = 1.0 - iou

    liou = (w_iou_c1 * per_circle[:, 0] + w_iou_c2 * per_circle[:, 1]).mean()

    return base + w_iou * liou
