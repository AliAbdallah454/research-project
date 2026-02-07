import math
import numpy as np
import torch

def circle_iou(c1: tuple[float, float, float], c2: tuple[float, float, float], eps=1e-12):
    """
    IoU of two circles.
    c1, c2: (cx, cy, r)
    returns: float
    """
    x0, y0, r0 = map(float, c1)
    x1, y1, r1 = map(float, c2)

    if r0 <= 0 or r1 <= 0:
        return 0.0

    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    a0 = math.pi * r0 * r0
    a1 = math.pi * r1 * r1

    # No overlap
    if d >= r0 + r1:
        return 0.0

    # One inside the other (including same center)
    if d <= abs(r0 - r1):
        inter = math.pi * min(r0, r1) ** 2
        union = a0 + a1 - inter
        return float(inter / max(union, eps))

    # Partial overlap
    # Clamp for numerical safety
    cos0 = (d*d + r0*r0 - r1*r1) / (2*d*r0)
    cos1 = (d*d + r1*r1 - r0*r0) / (2*d*r1)
    cos0 = min(1.0, max(-1.0, cos0))
    cos1 = min(1.0, max(-1.0, cos1))

    alpha = math.acos(cos0)
    beta  = math.acos(cos1)

    # Heron-like term for lens area
    k = (-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1)
    k = max(0.0, k)

    inter = r0*r0 * alpha + r1*r1 * beta - 0.5 * math.sqrt(k)
    union = a0 + a1 - inter
    return float(inter / max(union, eps))

def circle_iou_torch(c1: torch.Tensor, c2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Batched IoU of circles.

    Args:
        c1, c2: tensors with shape (..., 3) where last dim is (cx, cy, r)
                Shapes must be broadcastable (e.g., (B,3) vs (1,3), or (B,2,3) vs (B,2,3)).
        eps: small constant for numerical stability.

    Returns:
        iou: tensor with shape (...) (last dim removed), values in [0, 1].
    """
    c1 = torch.as_tensor(c1)
    c2 = torch.as_tensor(c2)
    dtype = torch.float32 if not c1.is_floating_point() else c1.dtype
    c1 = c1.to(dtype)
    c2 = c2.to(dtype)

    x0, y0, r0 = c1[..., 0], c1[..., 1], c1[..., 2]
    x1, y1, r1 = c2[..., 0], c2[..., 1], c2[..., 2]

    # invalid radii => IoU = 0
    invalid = (r0 <= 0) | (r1 <= 0)

    dx = x1 - x0
    dy = y1 - y0
    d = torch.sqrt(dx * dx + dy * dy + eps)

    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, dtype=dtype, device=c1.device)

    a0 = pi * r0 * r0
    a1 = pi * r1 * r1

    # Case masks
    no_overlap = d >= (r0 + r1)
    contained  = d <= torch.abs(r0 - r1)  # includes same center / full containment
    partial    = ~(no_overlap | contained | invalid)

    # Containment IoU
    rmin = torch.minimum(r0, r1)
    inter_cont = pi * rmin * rmin
    union_cont = a0 + a1 - inter_cont
    iou_cont = inter_cont / (union_cont + eps)

    # Partial overlap IoU
    # cos terms with safe denom + clamping
    denom0 = (2.0 * d * r0).clamp_min(eps)
    denom1 = (2.0 * d * r1).clamp_min(eps)

    cos0 = (d * d + r0 * r0 - r1 * r1) / denom0
    cos1 = (d * d + r1 * r1 - r0 * r0) / denom1
    cos0 = cos0.clamp(-1.0, 1.0)
    cos1 = cos1.clamp(-1.0, 1.0)

    alpha = torch.acos(cos0)
    beta  = torch.acos(cos1)

    k = (-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1)
    k = torch.clamp(k, min=0.0)

    inter_part = r0 * r0 * alpha + r1 * r1 * beta - 0.5 * torch.sqrt(k + eps)
    union_part = a0 + a1 - inter_part
    iou_part = inter_part / (union_part + eps)

    # Combine cases
    iou = torch.zeros_like(d)
    iou = torch.where(contained & ~invalid, iou_cont, iou)
    iou = torch.where(partial, iou_part, iou)
    # no_overlap and invalid remain 0

    return iou.clamp(0.0, 1.0)


if __name__ == "__main__":

    print(circle_iou((0.5, 0.6, 0.9), (0.5, 0.6, 0.9)))