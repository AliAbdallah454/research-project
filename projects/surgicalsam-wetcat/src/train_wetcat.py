import sys
import os
import os.path as osp
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset_wetcat import WetCatDataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses


print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="wetcat",
                    choices=["endovis_2018", "endovis_2017", "wetcat"],
                    help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3],
                    help='specify fold number for endovis_2017 dataset')
parser.add_argument("--debug", action="store_true", help="run 2 train + 2 val batches then exit")
parser.add_argument("--data_root", type=str, default=None,
                    help="Prepared dataset root (e.g. /data/wetcat_prepared_surgicalsam or C:\\...\\wetcat_prepared)")
parser.add_argument("--sam_ckpt", type=str, default=None,
                    help="Path to SAM checkpoint (.pth)")
parser.add_argument("--save_dir", type=str, default=None,
                    help="Output folder for logs/ckpts/plots")
parser.add_argument("--num_workers", type=int, default=0,
                    help="Dataloader workers (Windows: keep 0)")
parser.add_argument("--seq", type=str, default="seq1",
                    help="WetCat sequence folder name (default seq1)")
parser.add_argument("--val_ratio", type=float, default=0.2,
                    help="WetCat random split ratio (0.2 = 80/20)")
args = parser.parse_args()
debug = args.debug

print("======> Set Parameters for Training")
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 666

if args.data_root is not None:
    data_root_dir = args.data_root
else:
    if dataset_name == "wetcat":
        data_root_dir = "./data_backup/wetcat_prepared"
    else:
        data_root_dir = f"../data/{dataset_name}"

batch_size = 32
vit_mode = "h"

# set seed for reproducibility
random.seed(seed)
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Debug iteration limits
max_train_iters = 2 if debug else None
max_val_iters = 2 if debug else None

# -----------------------------
# Helpers for WetCat validation
# -----------------------------
def _safe_sigmoid(x):
    # In case preds are logits
    if x.min() < 0 or x.max() > 1:
        return torch.sigmoid(x)
    return x

@torch.no_grad()
def batch_iou_and_dice(preds, gts, thr=0.5, eps=1e-7):
    """
    preds: (B,1,H,W) float (logits or probs)
    gts:   (B,1,H,W) float in {0,1} (or 0..255)
    """
    preds = _safe_sigmoid(preds)
    if gts.max() > 1:
        gts = gts / 255.0

    p = (preds > thr).float()
    g = (gts > 0.5).float()

    p = p.view(p.size(0), -1)
    g = g.view(g.size(0), -1)

    inter = (p * g).sum(dim=1)
    union = (p + g - p * g).sum(dim=1)

    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (p.sum(dim=1) + g.sum(dim=1) + eps)
    return iou.mean().item(), dice.mean().item()


print("======> Load Dataset-Specific Parameters")
train_dataset = None
val_dataset = None
gt_endovis_masks = None

if "18" in dataset_name:
    num_tokens = 2
    val_dataset = Endovis18Dataset(data_root_dir=data_root_dir, mode="val", vit_mode="h")
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018/"

elif "17" in dataset_name:
    num_tokens = 4
    val_dataset = Endovis17Dataset(data_root_dir=data_root_dir, mode="val", fold=fold, vit_mode="h", version=0)
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val", fold=fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"

elif dataset_name == "wetcat":
    # WetCat is binary instrument segmentation => 1 class
    num_tokens = 2
    num_epochs = 20
    lr = 0.0005
    save_dir = "./work_dirs/wetcat/"
    if args.save_dir is not None:
        save_dir = args.save_dir
    gt_endovis_masks = None

    # Random 80/20 split on seq1
    full_dataset = WetCatDataset(data_root_dir=data_root_dir, vit_mode=vit_mode, seq=args.seq)
    val_ratio = args.val_ratio

    val_ratio = 0.2
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=g)

else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)


print("======> Load SAM")
if vit_mode == "h":
    sam_checkpoint = r"C:\Users\Admin\Desktop\MNproj\sam_vit_h_4b8939.pth"
    if args.sam_ckpt is not None:
        sam_checkpoint = args.sam_ckpt
model_type = f"vit_{vit_mode}"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cpu")

sam_prompt_encoder = sam.prompt_encoder
sam_decoder = sam.mask_decoder

for _, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for _, param in sam_decoder.named_parameters():
    param.requires_grad = True


print("======> Load Prototypes and Prototype-based Prompt Encoder")
num_classes = 1 if dataset_name == "wetcat" else 7
learnable_prototypes_model = Learnable_Prototypes(num_classes=num_classes, feat_dim=256).to("cpu")

protoype_prompt_encoder = Prototype_Prompt_Encoder(
    feat_dim=256,
    hidden_dim_dense=128,
    hidden_dim_sparse=128,
    size=64,
    num_tokens=num_tokens
).to("cpu")

with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {
        k.split("prompt_encoder.point_embeddings.")[-1]: v
        for k, v in state_dict.items()
        if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)
    }
    sam_pn_embeddings_weight_ckp = {
        "0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
        "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)
    }
    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

for _, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True

for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True


print("======> Define Optimiser and Loss")
seg_loss_model = DiceLoss().to("cpu")
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).to("cpu")

optimiser = torch.optim.Adam([
    {'params': learnable_prototypes_model.parameters()},
    {'params': protoype_prompt_encoder.parameters()},
    {'params': sam_decoder.parameters()}
], lr=lr, weight_decay=0.0001)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok=True)
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)


print("======> Start Training and Validation")
best_challenge_iou_val = -100.0
best_wetcat_iou = -1.0

train_losses = []
val_losses = []
val_ious = []
val_dices = []


for epoch in range(num_epochs):

    # choose the augmentation version to use for the current epoch
    if epoch % 2 == 0:
        version = 0
    else:
        version = int((epoch % 80 + 1) / 2)

    # Create train dataset per epoch for EndoVis (your original behavior)
    if "18" in dataset_name:
        train_dataset_epoch = Endovis18Dataset(data_root_dir=data_root_dir, mode="train", vit_mode=vit_mode, version=version)
    elif "17" in dataset_name:
        train_dataset_epoch = Endovis17Dataset(data_root_dir=data_root_dir, mode="train", fold=fold, vit_mode=vit_mode, version=version)
    else:
        # WetCat: use fixed random split dataset
        train_dataset_epoch = train_dataset

    train_dataloader = DataLoader(train_dataset_epoch, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # -----------------------------
    # Training
    # -----------------------------
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    epoch_loss_sum = 0.0
    epoch_steps = 0

    for it, (sam_feats, _, cls_ids, masks, class_embeddings) in enumerate(train_dataloader):
        sam_feats = sam_feats.to("cpu")
        cls_ids = cls_ids.to("cpu")
        masks = masks.to("cpu")
        class_embeddings = class_embeddings.to("cpu")

        prototypes = learnable_prototypes_model()

        preds, _ = model_forward_function(
            protoype_prompt_encoder, sam_prompt_encoder, sam_decoder,
            sam_feats, prototypes, cls_ids
        )

        proto_labels = torch.arange(1, prototypes.size(0) + 1, device=prototypes.device)
        contrastive_loss = contrastive_loss_model(
            prototypes, proto_labels,
            ref_emb=class_embeddings, ref_labels=cls_ids
        )
        seg_loss = seg_loss_model(preds, masks / 255)

        loss = seg_loss + contrastive_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss_sum += loss.item()
        epoch_steps += 1

        if it % 20 == 0:
            print_log(
                f"Epoch {epoch} | it {it} | "
                f"seg={seg_loss.item():.4f} "
                f"contr={contrastive_loss.item():.4f} "
                f"total={loss.item():.4f}",
                log_file
            )

        if debug and (it + 1) >= max_train_iters:
            break

    train_loss_epoch = epoch_loss_sum / max(1, epoch_steps)
    train_losses.append(train_loss_epoch)

    # -----------------------------
    # Validation
    # -----------------------------
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()

        if dataset_name != "wetcat":
            # EndoVis validation (your original)
            binary_masks = dict()
            for it, (sam_feats, mask_names, cls_ids, _, _) in enumerate(val_dataloader):
                sam_feats = sam_feats.to("cpu")
                cls_ids = cls_ids.to("cpu")

                preds, preds_quality = model_forward_function(
                    protoype_prompt_encoder, sam_prompt_encoder, sam_decoder,
                    sam_feats, prototypes, cls_ids
                )
                binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

                if debug and (it + 1) >= max_val_iters:
                    break

            endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
            endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

            print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results}", log_file)

            # (optional) store something for plotting; EndoVis doesn't compute loss here
            val_losses.append(np.nan)
            val_ious.append(endovis_results.get("challengIoU", np.nan))
            val_dices.append(np.nan)

            if endovis_results["challengIoU"] > best_challenge_iou_val:
                best_challenge_iou_val = endovis_results["challengIoU"]
                torch.save({
                    'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                    'sam_decoder_state_dict': sam_decoder.state_dict(),
                    'prototypes_state_dict': learnable_prototypes_model.state_dict(),
                }, osp.join(save_dir, 'model_ckp.pth'))
                print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)

        else:
            # WetCat validation: loss + IoU + Dice
            val_loss_sum = 0.0
            val_iou_sum = 0.0
            val_dice_sum = 0.0
            val_steps = 0

            for it, (sam_feats, _, cls_ids, masks, _) in enumerate(val_dataloader):
                sam_feats = sam_feats.to("cpu")
                cls_ids = cls_ids.to("cpu")
                masks = masks.to("cpu")

                preds, _ = model_forward_function(
                    protoype_prompt_encoder, sam_prompt_encoder, sam_decoder,
                    sam_feats, prototypes, cls_ids
                )

                seg_loss = seg_loss_model(preds, masks / 255)
                iou, dice = batch_iou_and_dice(preds, masks, thr=0.5)

                val_loss_sum += seg_loss.item()
                val_iou_sum += iou
                val_dice_sum += dice
                val_steps += 1

                if debug and (it + 1) >= max_val_iters:
                    break

            val_loss = val_loss_sum / max(1, val_steps)
            val_iou = val_iou_sum / max(1, val_steps)
            val_dice = val_dice_sum / max(1, val_steps)

            val_losses.append(val_loss)
            val_ious.append(val_iou)
            val_dices.append(val_dice)

            print_log(
                f"Validation - Epoch: {epoch}/{num_epochs-1}; "
                f"WetCat: train_loss={train_loss_epoch:.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}, val_dice={val_dice:.4f}",
                log_file
            )

            if val_iou > best_wetcat_iou:
                best_wetcat_iou = val_iou
                torch.save({
                    'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                    'sam_decoder_state_dict': sam_decoder.state_dict(),
                    'prototypes_state_dict': learnable_prototypes_model.state_dict(),
                }, osp.join(save_dir, 'model_ckp_best_wetcat.pth'))
                print_log(f"Best WetCat IoU: {best_wetcat_iou:.4f} at Epoch {epoch}", log_file)

            if epoch % 5 == 0:
                torch.save({
                    'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                    'sam_decoder_state_dict': sam_decoder.state_dict(),
                    'prototypes_state_dict': learnable_prototypes_model.state_dict(),
                }, osp.join(save_dir, f'model_ckp_epoch{epoch}.pth'))

    if debug:
        print("DEBUG run finished OK ✅")
        break


# -----------------------------
# Save plots
# -----------------------------
loss_plot_path = osp.join(save_dir, "loss_plot.png")
metrics_plot_path = osp.join(save_dir, "metrics_plot.png")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(loss_plot_path)
plt.close()

plt.figure()
plt.plot(val_ious, label="Val IoU")
plt.plot(val_dices, label="Val Dice")
plt.legend()
plt.title("Validation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig(metrics_plot_path)
plt.close()

print(f"Plots saved as:\n- {loss_plot_path}\n- {metrics_plot_path}")