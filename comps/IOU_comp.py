import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

print("cwd: ", os.getcwd())

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

model_path = f"./models/circle_regressor_ResNet18_v1.pt"

device = 'cpu'

model = CircleRegressor(pretrained=True)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

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
frames_path = os.path.join(par, "video_frames")

df = read_manual_results(manual_path)

r_ious = []
g_ious = []
model_r_ious = []
model_g_ious = []

w = 640
h = 360

for i in tqdm(range(len(df)), total=len(df), desc="Processing frames"):

    curr = df.iloc[i]

    time, r_gt, g_gt = get_gt_circles(curr)

    image_path = os.path.join(frames_path, f"img{time}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    r_model, g_model = predict_on_cv2_frames(model, img, transform=val_tf, device='cpu')
    _, r_class = detect_red_circle(img)
    g_class = (r_class[0], r_class[1], r_class[2] * 2.4)

    iou_r = circle_iou(r_gt, r_class)
    iou_g = circle_iou(g_gt, g_class)
    model_r_iou = circle_iou(r_gt, r_model)
    model_g_iou = circle_iou(g_gt, g_model)

    r_ious.append(iou_r)
    g_ious.append(iou_g)
    model_r_ious.append(model_r_iou)
    model_g_ious.append(model_g_iou)


ious = np.array(r_ious, dtype=float)
model_r_ious = np.array(model_r_ious, dtype=float)

x = np.arange(len(ious))

window = 25
kernel = np.ones(window) / window

smooth = np.convolve(ious, kernel, mode="same")
model_smooth = np.convolve(model_r_ious, kernel, mode="same")

alpha = 1

plt.figure(figsize=(10, 4))
plt.plot(x, r_ious, alpha=alpha, linewidth=1, label="IoU (raw) Red")
plt.plot(x, g_ious, alpha=alpha, linewidth=1, label="IoU (raw) Green")
plt.plot(x, model_r_ious, alpha=alpha, linewidth=1, label="IoU Model-ResNet18(raw) Red")
plt.plot(x, model_g_ious, alpha=alpha, linewidth=1, label="IoU Model-ResNet18(raw) Green")

# plt.plot(x, smooth, linewidth=2, label=f"IoU (rolling mean, w={window})")
# plt.plot(x, model_smooth, linewidth=2, label=f"IoU Model (rolling mean, w={window})")

plt.ylim(0, 1)
plt.xlabel("Time (frame/step)")
plt.ylabel("IoU")
plt.title("IoU over time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

graph_path = f'./plots/IOU_session{session}_participant{participant}.png'
plt.savefig(graph_path)

plt.show()
plt.close()
print("Saved to: ", graph_path)