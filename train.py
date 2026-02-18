import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

import time
from datetime import datetime
from tqdm import tqdm
import argparse

import os
import wandb

from src.architectures import CircleRegressorResNet
from src.loss_fn import circle_loss
from src.data import get_loaders
from eval.metrics import circle_iou_torch, center_error_torch


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
    p.add_argument("--device", default='cuda', help="Device that runs computations")

    p.add_argument("--testing", action="store_true", default=False, help="Indicate wehter file is being tested")

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
device = args.device
testing = args.testing

print(testing)

if os.path.exists(output_path):
    raise FileExistsError("Model already exists")

train_dl, val_dl, test_dl = get_loaders(root_path, batch_size=batch_size)

model = CircleRegressorResNet(backbone=resnet, pretrained=True, out_dim=6).to(device)

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

min_val_loss = float('inf')

print(f"Training Started {resnet} on {root_path}")
print(f"running for {epochs} epochs")

run = wandb.init(
    entity='alikhaledabdallah454-universit-paris-saclay',
    project='Research Project',
    name='circle-detection',
    config={
        "learning_rate": lr,
        "Optimizer": "Adam",
        "architecture": resnet,
        "backbone": resnet,
        "dataset": "gepromed",
        "epochs": epochs,
        "loss": {
            "w_center": w_center,
            "w_radius": w_radius,
            "w_iou": w_iou,
            "w_iou_c1": w_iou_c1,
            "w_iou_c2": w_iou_c2
        },
        "device": device,
    }
)

wandb.watch(model, log='gradients', log_freq=500)

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
        loss = circle_loss(preds, targets, w_center=w_center, w_radius=w_radius, w_iou=w_iou, w_iou_c1=w_iou_c1, w_iou_c2=w_iou_c2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        train_loss_sum += loss.item() * bs
        train_count += bs

        train_pbar.set_postfix(loss=f"{loss.item():.5f}")
        
        if testing:
            break

    avg_train_loss = train_loss_sum / max(1, train_count)

    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    val_pbar = tqdm(val_dl, total=len(val_dl), desc=f"Val   {epoch:02d}/{epochs}", leave=False)
    
    n = 0
    sum_iou1 = 0.0
    sum_iou2 = 0.0
    sum_c1 = 0.0
    sum_c2 = 0.0

    with torch.no_grad():
        for imgs, targets in val_pbar:
            imgs = imgs.to(device).float()
            targets = targets.to(device).float()

            preds = model(imgs)
            loss = circle_loss(preds, targets)

            bs = imgs.size(0)
            val_loss_sum += loss.item() * bs
            val_count += bs

            val_pbar.set_postfix(loss=f"{loss.item():.5f}")

            ### IOU Eval
            ious = circle_iou_torch(preds, targets)
            n += ious.shape[0]
            
            sum_iou1 += ious[:, 0].sum().item()
            sum_iou2 += ious[:, 1].sum().item()

            ### Center Error
            center_errors = center_error_torch(preds, targets)
            
            sum_c1 += center_errors[:, 0].sum().item()
            sum_c2 += center_errors[:, 1].sum().item()

            if testing:
                break
    
    mean_iou1 = sum_iou1 / max(n, 1)
    mean_iou2 = sum_iou2 / max(n, 1)

    mean_c1 = sum_c1 / max(n, 1)
    mean_c2 = sum_c2 / max(n, 1)

    avg_val_loss = val_loss_sum / max(1, val_count)

    epoch_time = time.time() - epoch_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if avg_val_loss < min_val_loss:
        torch.save(model.state_dict(), output_path)
        min_val_loss = avg_val_loss

    wandb.log({
        "epoch": epoch,
        "lr": lr,
        "train/loss": avg_train_loss,
        "val/loss": avg_val_loss,
        "val/best_loss": min_val_loss,
        "metrics/iou1": mean_iou1,
        "metrics/iou2": mean_iou2,
        "metrics/center-error1": mean_c1,
        "metrics/center-error2": mean_c2
    })

    print(
        f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
        f"| Time: {epoch_time:.2f}s | Finished at: {finished_at}"
    )