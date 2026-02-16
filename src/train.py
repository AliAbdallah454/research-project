import torch
import torch.nn.functional as F
import torch.optim as optim

import time
from datetime import datetime
from tqdm import tqdm
import argparse

import os
from itertools import islice

from src.architectures import CircleRegressorResNet
from src.data import get_loaders
from eval.metrics import circle_iou_torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resnet", required=True, help="ResNet type")
    p.add_argument("--data", required=True, help="Dataset root or config path")
    p.add_argument("--batch", type=int, required=True, help="Batch size")
    p.add_argument("--output", required=True, help="Output path for model")
    p.add_argument("--epochs", type=int, required=True, help="Number of training epochs")

    p.add_argument("--w-center", type=float, default=1.0, help="Penalty of center loss")
    p.add_argument("--w-radius", type=float, default=2.0, help="Penalty of radius loss")
    p.add_argument("--w-iou", type=float, default=0.0, help="Penalty of IOU")
    p.add_argument("--w-iou-c1", type=float, default=1.0, help="Penalty of IOU circle1")
    p.add_argument("--w-iou-c2", type=float, default=1.0, help="Penalty of IOU circle2")

    return p.parse_args()

args = parse_args()

root_path = args.data
resnet = args.resnet
batch_size = args.batch
output_path = args.output
epochs = args.epochs
w_center = args.w_center
w_radius = args.w_radius
w_iou = args.w_iou
w_iou_c1 = args.w_iou_c1
w_iou_c2 = args.w_iou_c2

if os.path.exists(output_path):
    raise FileExistsError("Model already exists")

train_dl, val_dl, test_dl = get_loaders(root_path, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is: {device}")

model = CircleRegressorResNet(backbone=resnet, pretrained=True, out_dim=6).to(device)

def circle_loss(
    preds,
    targets,
    w_center=1.0,
    w_radius=2.0,
    w_iou=1.0,
    w_iou_c1=1.0,
    w_iou_c2=1.0,
    beta=0.02,
    iou_mode="one_minus",
    eps=1e-7,
):
    """
    preds/targets: (B, 6) = [cx1, cy1, r1, cx2, cy2, r2], normalized to [0,1]
    """

    preds_v = preds.view(-1, 2, 3)
    targets_v = targets.view(-1, 2, 3)

    pc, pr = preds_v[..., :2], preds_v[..., 2]
    tc, tr = targets_v[..., :2], targets_v[..., 2]

    lc = F.smooth_l1_loss(pc, tc, beta=beta, reduction="mean")
    lr = F.smooth_l1_loss(pr, tr, beta=beta, reduction="mean")

    base = w_center * lc + w_radius * lr

    if w_iou == 0.0:
        return base

    iou = circle_iou_torch(preds, targets)

    if iou_mode == "log":
        per_circle = -torch.log(iou.clamp_min(eps))
    else:
        per_circle = 1.0 - iou

    liou = (w_iou_c1 * per_circle[:, 0] + w_iou_c2 * per_circle[:, 1]).mean()

    return base + w_iou * liou


optimizer = optim.Adam(model.parameters(), lr=1e-3)

LOSS_FN = circle_loss

min_val_loss = float('inf')

print(f"Training Started {resnet} on {root_path}")
print(f"running for {epochs} epochs")
for epoch in range(1, epochs + 1):

    epoch_start = time.time()

    model.train()
    train_loss_sum = 0.0
    train_count = 0

    train_pbar = tqdm(train_dl, total=len(train_dl), desc=f"Train {epoch:02d}/{epochs}", leave=False)
    for imgs, targets in train_pbar:
        imgs = imgs.to(device).float()
        targets = targets.to(device).float()

        preds = model(imgs)
        loss = LOSS_FN(preds, targets, w_center=w_center, w_radius=w_radius, w_iou=w_iou, w_iou_c1=w_iou_c1, w_iou_c2=w_iou_c2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        train_loss_sum += loss.item() * bs
        train_count += bs

        train_pbar.set_postfix(loss=f"{loss.item():.5f}")

    avg_train_loss = train_loss_sum / max(1, train_count)

    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    val_pbar = tqdm(val_dl, total=len(val_dl), desc=f"Val   {epoch:02d}/{epochs}", leave=False)
    with torch.no_grad():
        for imgs, targets in val_pbar:
            imgs = imgs.to(device).float()
            targets = targets.to(device).float()

            preds = model(imgs)
            loss = LOSS_FN(preds, targets)

            bs = imgs.size(0)
            val_loss_sum += loss.item() * bs
            val_count += bs

            val_pbar.set_postfix(loss=f"{loss.item():.5f}")

    avg_val_loss = val_loss_sum / max(1, val_count)

    epoch_time = time.time() - epoch_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if avg_val_loss < min_val_loss:
        torch.save(model.state_dict(), output_path)
        min_val_loss = avg_val_loss

    print(
        f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
        f"| Time: {epoch_time:.2f}s | Finished at: {finished_at}"
    )

torch.save(model.state_dict(), output_path)