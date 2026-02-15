import math
import numpy as np
import torch

from typing import Tuple

Circle = Tuple[float, float, float]

def circle_iou(c1: Circle, c2: Circle, eps=1e-12):
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

def center_error_norm(pred: Circle, gt: Circle, eps: float = 1e-12) -> float:
    """
    Center error for normalized circles.
    Returns Euclidean distance in normalized units (0..~sqrt(2)).
    """
    px, py, _ = map(float, pred)
    gx, gy, _ = map(float, gt)
    return math.hypot(px - gx, py - gy)

def temporal_stability_norm(prev: Circle, curr: Circle, radius_weight: float = 1.0, eps: float = 1e-12) -> float:
    """
    Temporal stability (jitter) between consecutive circles (usually predictions),
    in normalized units.

    jitter = sqrt( dx^2 + dy^2 + (radius_weight * dr)^2 )
    """
    x0, y0, r0 = map(float, prev)
    x1, y1, r1 = map(float, curr)

    dx = x1 - x0
    dy = y1 - y0
    dr = r1 - r0

    return math.sqrt(dx*dx + dy*dy + (radius_weight * dr) * (radius_weight * dr))

if __name__ == "__main__":

    print(circle_iou((0.5, 0.6, 0.9), (0.5, 0.6, 0.9)))