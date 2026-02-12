import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from helpers import read_manual_results, predict_on_cv2_frames
from helpers import get_gt_circles

import torch
from architectures import CircleRegressorResNet
import torchvision.transforms as T

from metrics import temporal_stability_norm
from classical_methods.red_circle_detection import detect_red_circle
from tqdm import tqdm

from typing import Tuple

import os
import argparse

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

model_path = f"./models/circle_regressor_ResNet18_v1.pt"

device = 'cpu'

model = CircleRegressorResNet(backbone='resnet18', pretrained=True)
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

ious = []
model_ious = []

for i in tqdm(range(len(df)), total=len(df), desc="Processing frames"):

    curr = df.iloc[i]

    time, r_gt, g_gt = get_gt_circles(curr)

    image_path = os.path.join(frames_path, f"img{time}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    r_model, g_model = predict_on_cv2_frames(model, img, transform=val_tf, device='cpu')
    _, r_class = detect_red_circle(img)

    iou = temporal_stability_norm(r_gt, r_class)
    model_iou = temporal_stability_norm(r_gt, r_model)

    ious.append(iou)
    model_ious.append(model_iou)


ious = np.array(ious, dtype=float)
model_ious = np.array(model_ious, dtype=float)

x = np.arange(len(ious))

alpha = 1

plt.figure(figsize=(10, 4))
plt.plot(x, ious, alpha=alpha, linewidth=1, label="Temporal Stability")
plt.plot(x, model_ious, alpha=alpha, linewidth=1, label="Temporal Stability Model-ResNet18(raw)")

plt.xlabel("Time (frame/step)")
plt.ylabel("Stability")
plt.title("Stability over time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

graph_path = f'./plots/TEMPORAL_STABILITY_session{session}_participant{participant}.png'
plt.savefig(graph_path)

plt.show()
plt.close()
print("Saved to: ", graph_path)