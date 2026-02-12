import torch
from architectures import CircleRegressorResNet
import torchvision.transforms as T

import cv2

import argparse
from typing import Tuple

from helpers import predict_on_cv2_frames

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

model_path = f"./models/circle_regressor_ResNet50.pt"
# model_path = f"./models/circle_regressor_ResNet18_v1.pt"

device = 'cpu'

model = CircleRegressorResNet(backbone='resnet50', pretrained=True)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

def build_args():
    p = argparse.ArgumentParser(description="Run CircleRegressor on one frame.")
    p.add_argument("--session", type=int, required=True, help="Session number, e.g. 2")
    p.add_argument("--participant", type=int, required=True, help="Participant number, e.g. 14")

    return p.parse_args()

args = build_args()

def denorm_circle(c, w, h):
    x = int(round(c[0] * w))
    y = int(round(c[1] * h))
    r = int(round(c[2] * min(w, h)))

    return x, y, r

video_path = f"./data/Cornia/Session{args.session}_Light/Participant{args.participant}/video_640x360.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    r_c, g_c = predict_on_cv2_frames(model, frame, val_tf)

    h, w, _ = frame.shape

    x, y, r = denorm_circle(r_c, w, h)
    cv2.circle(frame, (x, y), r, (0, 0, 255), 3)
    cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

    x, y, r = denorm_circle(g_c, w, h)
    cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
    cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()