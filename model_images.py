from architectures import CircleRegressor
import torch
import torchvision.transforms as T

from architectures import CircleRegressor

from helpers import draw_two_circles_on_pil
from PIL import Image

import os
import argparse

def build_args():
    p = argparse.ArgumentParser(description="Run CircleRegressor on one frame.")
    p.add_argument("--session", type=int, required=True, help="Session number, e.g. 2")
    p.add_argument("--participant", type=int, required=True, help="Participant number, e.g. 14")
    p.add_argument("--frame", type=int, required=True, help="Frame index/number, e.g. 100")

    return p.parse_args()

args = build_args()

model_path = f"./models/circle_regressor_v1.pt"

device = 'cpu'

model = CircleRegressor(True)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval();

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

session = args.session
participant = args.participant
fram_nb = args.frame

img = f"./data/processed_data/Session{session}_Light/Participant{participant}/video_frames/img{fram_nb}.jpg"

if not os.path.exists(img):
    raise FileNotFoundError("Image not found")

img = Image.open(img)

out = model(val_tf(img).unsqueeze(0))
ts = tuple(out.detach().squeeze(0).tolist())
print(ts)

draw_two_circles_on_pil(img, ts)