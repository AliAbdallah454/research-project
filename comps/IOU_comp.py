import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

import torch
import torchvision.transforms as T

import argparse

from tqdm import tqdm

from classical_methods.red_circle_detection import detect_red_circle
from src.metrics import circle_iou
from src.architectures import CircleRegressorResNet
from src.helpers import get_gt_circles, read_manual_results, predict_on_cv2_frames


print("cwd: ", os.getcwd())

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
    p.add_argument("--models", nargs='+', required=True, help="one or more model paths")

    return p.parse_args()

args = build_args()

session = args.session
participant = args.participant
model_paths = args.models

device = 'cpu'

def load_model(model_path: str) -> CircleRegressorResNet:

    if 'resnet18' in model_path or 'ResNet18' in model_path:
        model = CircleRegressorResNet(backbone='resnet18', pretrained=True)
    elif 'resnet34' in model_path:
        model = CircleRegressorResNet(backbone='resnet34', pretrained=True)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, model_path.split('.')[-2]

class Mod:

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.r_ious = []
        self.g_ious = []

mods: List[Mod] = []

for model_path in args.models:
    print("Loading: ", model_path)
    model, name = load_model(model_path)
    print(name)
    mods.append(Mod(model, name))

par = f"./data/processed_data/Session{session}_Light/Participant{participant}"

manual_path = os.path.join(par, "normalized_results_manual.txt")
frames_path = os.path.join(par, "video_frames")

df = read_manual_results(manual_path)

w = 640
h = 360

for i in tqdm(range(len(df)), total=len(df), desc="Processing frames"):

    curr = df.iloc[i]

    time, r_gt, g_gt = get_gt_circles(curr)

    image_path = os.path.join(frames_path, f"img{time}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for mod in mods:
        r, g = predict_on_cv2_frames(mod.model, img, transform=val_tf, device='cpu')
        mod.r_ious.append(circle_iou(r_gt, r))
        mod.g_ious.append(circle_iou(g_gt, g))


x = np.arange(len(mods[0].r_ious))

window = 25
kernel = np.ones(window) / window

alpha = 1

plt.figure(figsize=(10, 4))

for mod in mods:
    plt.plot(x, mod.r_ious, alpha=alpha, linewidth=1, label=f"IoU {mod.name} Red")
    plt.plot(x, mod.g_ious, alpha=alpha, linewidth=1, label=f"IoU {mod.name} Green")

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
# print("Saved to: ", graph_path)