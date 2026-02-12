import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch

from typing import Tuple, Callable

def draw_two_circles_on_pil(
    pil_img: Image.Image,
    params: tuple,
    out_path: str | None = None,
    thickness: int = 1,
    figsize=(8, 8),
    return_img: bool = False,
):
    """
    Draw two circles on a PIL image and display using matplotlib.

    Args:
        pil_img: PIL.Image.Image
        params: (xc1, yc1, r1, xc2, yc2, r2) in pixel coordinates
        out_path: optional save path (saves a PNG/JPG with circles)
        thickness: circle line thickness
        figsize: matplotlib figure size
        return_img: if True, return the result as a PIL Image; otherwise return None

    Notes:
        - Circle 1 = RED, Circle 2 = GREEN
        - No array printed under the figure (returns None by default)
    """
    if len(params) != 6:
        raise ValueError("params must be (xc1, yc1, r1, xc2, yc2, r2)")

    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    W, H = pil_img.size

    xc1, yc1, r1, xc2, yc2, r2 = params

    xc1 = int(round(xc1 * W))
    yc1 = int(round(yc1 * H))
    r1 = int(round(r1 * min(W, H)))

    xc2 = int(round(xc2 * W))
    yc2 = int(round(yc2 * H))
    r2 = int(round(r2 * min(W, H)))

    xc1, yc1, r1 = int(round(xc1)), int(round(yc1)), int(round(r1))
    xc2, yc2, r2 = int(round(xc2)), int(round(yc2)), int(round(r2))

    drawn = bgr.copy()

    cv2.circle(drawn, (xc1, yc1), r1, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    cv2.circle(drawn, (xc1, yc1), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    cv2.circle(drawn, (xc2, yc2), r2, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    cv2.circle(drawn, (xc2, yc2), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    drawn_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(drawn_rgb)
    plt.axis("off")
    plt.show()

    if out_path is not None:
        Image.fromarray(drawn_rgb).save(out_path)

    if return_img:
        return Image.fromarray(drawn_rgb)
    return None

def read_manual_results(path: str):

    expected_cols = [
        "Time in s",
        "Defined zone",
        "Point number",
        "Center-X Zone 1",
        "Center-Y Zone 1",
        "Radius Zone 1",
        "Center-X Zone 2",
        "Center-Y Zone 2",
        "Radius Zone 2"
    ]

    df = pd.read_csv(path, sep='\t', usecols=expected_cols)
    return df

def get_gt_circles(df_instance) -> Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]:

    curr = df_instance

    time = int(curr['Time in s'])
    x1 = float(curr['Center-X Zone 1'])
    y1 = float(curr['Center-Y Zone 1'])
    r1 = float(curr['Radius Zone 1'])

    x2 = float(curr['Center-X Zone 2'])
    y2 = float(curr['Center-Y Zone 2'])
    r2 = float(curr['Radius Zone 2'])

    return time, (x1, y1, r1), (x2, y2, r2)

@torch.inference_mode()
def predict_on_cv2_frames(model, frame: np.ndarray, transform: Callable, device: str='cpu', verbose: bool=False) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:

    is_training = model.training
    model.eval()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    x = transform(frame_pil)
    x = x.unsqueeze(0).to(device)

    out = model(x)
    out = out.squeeze(0).detach().cpu().tolist()

    r_pred = tuple(out[:3])
    g_pred = tuple(out[3:])

    if is_training: model.train()

    return r_pred, g_pred