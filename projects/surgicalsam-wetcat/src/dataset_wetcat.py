import torch 
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import os
import os.path as osp
import cv2
import re

class WetCatDataset(Dataset):
    def __init__(self, data_root_dir, vit_mode = "h", seq = "seq1", require_precomputed = True ):
        self.vit_mode = vit_mode
        self.seq = seq
        self.require_precomputed = require_precomputed
        base = osp.join(data_root_dir, "0")
        self.img_dir = osp.join(base, "images", seq)
        self.masks_dir = osp.join(base, "binary_annotations", seq)
        self.feat_dr = osp.join(base, f"sam_features_{vit_mode}", seq)
        self.emb_dr = osp.join(base, f"class_embeddings_{vit_mode}", seq)

        if not osp.isdir(self.img_dir):
            raise FileNotFoundError(f"missing img_dir:{self.img_dir}")
        if not osp.isdir(self.masks_dir):
            raise FileNotFoundError(f"missing mask_dir:{self.masks_dir}")
        

        self.items= []
        for mask_file in sorted(os.listdir(self.masks_dir)):
            if not mask_file.lower().endswith(".png"):
                continue
        
            m= re.search(r"_class(\d+)\.png$", mask_file, flags= re.IGNORECASE)
            if m is None:
                continue
            cls_id = 1

            frame_stem = re.sub(r"_class\d+$", "", osp.splitext(mask_file)[0], flags=re.IGNORECASE)
            frame_file = frame_stem + ".jpg"

            img_path = osp.join(self.img_dir, frame_file)
            mask_path = osp.join(self.masks_dir, mask_file)

            feat_path = osp.join(self.feat_dr, frame_stem + ".npy")
            emb_path = osp.join(self.emb_dr, mask_file.replace(".png", ".npy"))

            if not osp.exists(img_path):
                continue
            if require_precomputed and (not osp.exists(feat_path) or not osp.exists(emb_path)):
                continue  

            self.items.append((img_path, mask_path, feat_path, emb_path, cls_id, mask_file))

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, mask_path, feat_path, emb_path, cls_id, mask_file = self.items[idx]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1280, 1024), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

        sam_feat = np.load(feat_path) if osp.exists(feat_path) else None
        class_embedding = np.load(emb_path) if osp.exists(emb_path) else None

        rel_mask = f"{self.seq}/{osp.basename(mask_path)}"
        return sam_feat, rel_mask, cls_id, mask, class_embedding





