#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:46:49 2026

@author: djayadeep

"""

''' 
Clone SAM2:
pip install --no-cache-dir -U git+https://github.com/facebookresearch/segment-anything-2.git
Create checkpoint directory
mkdir -p /mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints
Download pretrained model
wget -P /mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
'''
import torch
import numpy as np
import cv2
import os
import argparse
# using video predictor
from sam2.build_sam import build_sam2_video_predictor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", default="/mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints/sam2_hiera_large.pt", 
                   help="SAM2 weights path")
    p.add_argument("--frames-path", required=True, help="Directory containing {i}.jpg frames")
    p.add_argument("--out-path", required=True, help="Directory where masks will be stored")

    return p.parse_args()

args = parse_args()

#  cofig folders
CHECKPOINT = args.checkpoint_path
MODEL_CFG = "sam2_hiera_l.yaml"
FRAME_DIR = args.frames_path
BASE_OUTPUT_DIR = args.out_path

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
# sub directories
DIRS = {
    "overlays": os.path.join(BASE_OUTPUT_DIR, "overlays"),
    "rgb_masks": os.path.join(BASE_OUTPUT_DIR, "rgb_masks"),
    "yellow_masks": os.path.join(BASE_OUTPUT_DIR, "yellow_masks"),
    "combined_masks": os.path.join(BASE_OUTPUT_DIR, "combined_rgby_masks")
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

# overlay colours
colors = {
    1: [0, 255, 0],    # Green
    2: [255, 0, 0],    # Blue
    3: [0, 0, 255],    # Red
    4: [0, 255, 255]   # Yellow
}

# gpu initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device)
inference_state = predictor.init_state(video_path=FRAME_DIR)

frame_names = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(('.jpg', '.png'))])

# "ANCHOR FRAME " initialisation 
ANCHOR_FRAME_IDX = 23

# Identify tools to segment
prompt_schedule = [
    {"name": "Left Tool", "obj_id": 1, "box": [0, 0, 1050, 600]},
    {"name": "Right Tool", "obj_id": 2, "box": [1200, 250, 1920, 500]},
    {"name": "Needle", "obj_id": 3, "box": [1150, 300, 1300, 450]},
    {"name": "Surgeon Hand", "obj_id": 4, "box": [0, 0, 550, 900]}
]

predictor.reset_state(inference_state)

for obj in prompt_schedule:
    print(f" looking for {obj['name']}")
    box = np.array(obj["box"], dtype=np.float32)
    predictor.add_new_points_or_box(
        inference_state, 
        frame_idx=ANCHOR_FRAME_IDX, 
        obj_id=obj["obj_id"], 
        box=box
    )

video_segments = {} 

print("tracking through video")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Backward pass from anchor frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ANCHOR_FRAME_IDX, reverse=True):
        video_segments[out_frame_idx] = (out_obj_ids, out_mask_logits)
    
    # Forward pass from anchoe frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ANCHOR_FRAME_IDX, reverse=False):
        video_segments[out_frame_idx] = (out_obj_ids, out_mask_logits)

# save to directories
print("Saving outputs to folders:")
for out_frame_idx in sorted(video_segments.keys()):
    out_obj_ids, out_mask_logits = video_segments[out_frame_idx]
    img_name = frame_names[out_frame_idx]
    
    frame_path = os.path.join(FRAME_DIR, img_name)
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]

    # creating mask and overlay
    rgb_mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    yellow_mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    combined_mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_canvas = frame.copy()

    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        
        if mask.any():
            color = colors.get(obj_id, [255, 255, 255])            
            combined_mask_canvas[mask] = color            
            overlay_canvas[mask] = color
            
            # separate instruments from hand
            if obj_id in [1, 2, 3]: # Left instrument Right instrument, Needle
                rgb_mask_canvas[mask] = color
            elif obj_id == 4:       # Surgeon Hand
                yellow_mask_canvas[mask] = color

    cv2.addWeighted(overlay_canvas, 0.4, frame, 0.6, 0, overlay_canvas)

    # Save to respective folders
    cv2.imwrite(os.path.join(DIRS["overlays"], f"overlay_{img_name}"), overlay_canvas)
    cv2.imwrite(os.path.join(DIRS["rgb_masks"], f"rgb_{img_name}"), rgb_mask_canvas)
    cv2.imwrite(os.path.join(DIRS["yellow_masks"], f"yellow_{img_name}"), yellow_mask_canvas)
    cv2.imwrite(os.path.join(DIRS["combined_masks"], f"combined_{img_name}"), combined_mask_canvas)
