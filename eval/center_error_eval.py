import torch

import argparse
from tqdm import tqdm

from src.data import get_loaders
from .metrics import center_error_torch
from src.architectures import CircleRegressorResNet

def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to the model to be evaluated")
    p.add_argument("--resnet", required=True, help="ResNet used")
    p.add_argument("--set", required=True, choices=["train", "val", "test"], help="Set to be evaluated on")
    p.add_argument("--data", required=True, help="Path to data")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--device", default='cuda', help="Select device for torch")

    return p.parse_args()

args = parse_args()

model_path = args.model_path
resnet = args.resnet
eval_set = args.set
data_path = args.data
batch_size = args.batch
device = args.device

model = CircleRegressorResNet(resnet, True).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

train_dl, val_dl, test_dl = get_loaders(data_path, batch_size)
loader = {"train": train_dl, "val": val_dl, "test": test_dl}[eval_set]

n = 0
sum_c1 = 0.0
sum_c2 = 0.0

with torch.no_grad():
    for imgs, targets in tqdm(loader, desc="IoU Eval", unit="Batch"):

        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)

        center_errors = center_error_torch(preds, targets)

        n += center_errors.shape[0]
        
        sum_c1 += center_errors[:, 0].sum().item()
        sum_c2 += center_errors[:, 1].sum().item()

mean_c1 = sum_c1 / max(n, 1)
mean_c2 = sum_c2 / max(n, 1)
mean_all  = (mean_c1 + mean_c2) / 2.0

print("mean_center_error_circle1 =", mean_c1)
print("mean_center_error_circle2 =", mean_c2)
print("mean_center_error_overall =", mean_all)