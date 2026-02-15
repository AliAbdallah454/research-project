import torch

def circle_iou_torch(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute IoU for two circles per sample.

    Args:
        preds:   Tensor [B, 6] = (x1, y1, r1, x2, y2, r2)
        targets: Tensor [B, 6] = (x1, y1, r1, x2, y2, r2)
        eps: small constant

    Returns:
        ious: Tensor [B, 2] where:
              ious[:,0] = IoU of first circle (preds[:,:3] vs targets[:,:3])
              ious[:,1] = IoU of second circle (preds[:,3:] vs targets[:,3:])
    """
    preds = torch.as_tensor(preds)
    targets = torch.as_tensor(targets)

    if preds.ndim != 2 or preds.size(-1) != 6:
        raise ValueError(f"preds must be [B,6], got {tuple(preds.shape)}")
    if targets.ndim != 2 or targets.size(-1) != 6:
        raise ValueError(f"targets must be [B,6], got {tuple(targets.shape)}")

    # Keep device, ensure float
    device = preds.device
    dtype = preds.dtype if preds.is_floating_point() else torch.float32
    preds = preds.to(dtype=dtype)
    targets = targets.to(dtype=dtype, device=device)

    # reshape to [B,2,3]
    p = preds.view(-1, 2, 3)
    t = targets.view(-1, 2, 3)

    # --- IoU core (same math as your circle_iou_torch, now operating on [B,2,3]) ---
    x0, y0, r0 = p[..., 0], p[..., 1], p[..., 2]
    x1, y1, r1 = t[..., 0], t[..., 1], t[..., 2]

    invalid = (r0 <= 0) | (r1 <= 0)

    dx = x1 - x0
    dy = y1 - y0
    d = torch.sqrt(dx * dx + dy * dy + eps)

    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, dtype=dtype, device=device)

    a0 = pi * r0 * r0
    a1 = pi * r1 * r1

    no_overlap = d >= (r0 + r1)
    contained = d <= torch.abs(r0 - r1)
    partial = ~(no_overlap | contained | invalid)

    rmin = torch.minimum(r0, r1)
    inter_cont = pi * rmin * rmin
    union_cont = a0 + a1 - inter_cont
    iou_cont = inter_cont / (union_cont + eps)

    denom0 = (2.0 * d * r0).clamp_min(eps)
    denom1 = (2.0 * d * r1).clamp_min(eps)

    cos0 = (d * d + r0 * r0 - r1 * r1) / denom0
    cos1 = (d * d + r1 * r1 - r0 * r0) / denom1
    cos0 = cos0.clamp(-1.0, 1.0)
    cos1 = cos1.clamp(-1.0, 1.0)

    alpha = torch.acos(cos0)
    beta = torch.acos(cos1)

    k = (-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1)
    k = torch.clamp(k, min=0.0)

    inter_part = r0 * r0 * alpha + r1 * r1 * beta - 0.5 * torch.sqrt(k + eps)
    union_part = a0 + a1 - inter_part
    iou_part = inter_part / (union_part + eps)

    iou = torch.zeros_like(d)
    iou = torch.where(contained & ~invalid, iou_cont, iou)
    iou = torch.where(partial, iou_part, iou)

    return iou.clamp(0.0, 1.0)  # [B,2]

def center_error_torch(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    preds, targets: [B,6]
    returns: [B,2] center error for circle1 and circle2
    """
    preds = torch.as_tensor(preds).float()
    targets = torch.as_tensor(targets).float().to(preds.device)

    dx1 = preds[:, 0] - targets[:, 0]
    dy1 = preds[:, 1] - targets[:, 1]
    dx2 = preds[:, 3] - targets[:, 3]
    dy2 = preds[:, 4] - targets[:, 4]

    e1 = torch.sqrt(dx1 * dx1 + dy1 * dy1 + eps)
    e2 = torch.sqrt(dx2 * dx2 + dy2 * dy2 + eps)
    return torch.stack([e1, e2], dim=1)  # [B,2]
