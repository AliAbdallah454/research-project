import os, os.path as osp, re
import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np 
from segment_anything.utils.transforms import ResizeLongestSide
import argparse


def preprocess(x):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1,1,1)
    x = (x-pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, 1024-w, 0, 1024-h))
    return x

def set_mask(mask_rgb):
     mask = ResizeLongestSide(1024).apply_image(mask_rgb)
     mask = torch.as_tensor(mask).permute(2,0,1)[None]
     return preprocess(mask)


def mask_to_frame(mask_name):
    return re.sub(r"_class\d+\.png$", "", mask_name, flags=re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_ckpt", required=True)
    parser.add_argument("--vit_mode", default="h", choices=["h","l","b"])
    args = parser.parse_args()

    base = osp.join(args.data_root, "0")
    img_dir  = osp.join(base, "images", "seq1")
    mask_dir = osp.join(base, "binary_annotations", "seq1")

    feat_dir = osp.join(base, f"sam_features_{args.vit_mode}", "seq1")
    emb_dir  = osp.join(base, f"class_embeddings_{args.vit_mode}", "seq1")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    sam = sam_model_registry[f"vit_{args.vit_mode}"](checkpoint=args.sam_ckpt)
    sam.cuda()
    predictor = SamPredictor(sam)

    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    masks  = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    masks_by_frame = {}
    for m in masks:
        masks_by_frame.setdefault(mask_to_frame(m), []).append(m)

    for i, frame_name in enumerate(frames):
        stem = osp.splitext(frame_name)[0]
        if stem not in masks_by_frame:
            continue

        print(f"[{i+1}/{len(frames)}] {frame_name}")

        img = cv2.imread(osp.join(img_dir, frame_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        predictor.set_image(img)
        feat = predictor.features.squeeze().permute(1,2,0).cuda().numpy()
        np.save(osp.join(feat_dir, stem + ".npy"), feat)

        for mask_name in masks_by_frame[stem]:
            mask = cv2.imread(osp.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            mask = (mask == 255).astype(np.uint8) * 255
            rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], -1)

            m = set_mask(rgb).cuda()
            m = F.interpolate(m, size=(64,64), mode="bilinear")
            m = m.squeeze()[0].cuda().numpy()

            if not (m > 0).any():
                continue

            emb = feat[m > 0].mean(0)
            np.save(osp.join(emb_dir, mask_name.replace(".png",".npy")), emb)

    print("Done.")

if __name__ == "__main__":
    main()