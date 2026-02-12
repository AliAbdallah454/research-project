import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from helpers import read_manual_results, predict_on_cv2_frames

import torch
from architectures import CircleRegressor
import torchvision.transforms as T

from metrics import circle_iou
from red_circle_detection import detect_red_circle
from tqdm import tqdm
from helpers import get_gt_circles

from typing import Tuple

import os
import argparse

import math

def build_args():
    p = argparse.ArgumentParser(description="Run CircleRegressor on one frame.")
    p.add_argument("--session", type=int, required=True, help="Session number, e.g. 2")
    p.add_argument("--participant", type=int, required=True, help="Participant number, e.g. 14")

    return p.parse_args()

args = build_args()

session = args.session
participant = args.participant


par = f"./data/processed_data/Session{session}_Light/Participant{participant}"

manual_path = os.path.join(par, "normalized_results_manual.txt")

df = read_manual_results(manual_path)

w = 640
h = 360

diffs = []
xs = []
ys = []

for i in tqdm(range(len(df)), total=len(df), desc="Processing frames"):

    curr = df.iloc[i]

    time, r_gt, g_gt = get_gt_circles(curr)

    x1, y1, r1 = r_gt
    x2, y2, r2 = g_gt

    diffs.append(r2 / r1)
    xs.append(abs(x1 - x2))
    ys.append(abs(y1 - y2))

alpha = 0.5
x = np.arange(len(diffs))

maximum = max(diffs)
minimum = min(diffs)

plt.figure(figsize=(10, 4))
plt.plot(x, diffs)

plt.text(
    0.98, 0.98,
    f"max: {maximum:.3f}\nmin: {minimum:.3f}\nmargin: {(maximum - minimum):.3f}",
    transform=plt.gca().transAxes,
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.8)
)

plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(x, xs)
plt.plot(x, ys)
plt.tight_layout()
plt.show()
plt.close()