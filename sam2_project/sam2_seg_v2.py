#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: djayadeep
"""
#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import os
import json
import argparse
import sys
from sam2.build_sam import build_sam2_video_predictor

def parse_args():
    p = argparse.ArgumentParser(description="SAM2 Video Processor")
    p.add_argument("--checkpoint-path", 
                   default="/mnt/data/wetcat_dataset/wetcat_code/sam2_project/checkpoints/sam2_hiera_large.pt", 
                   help="SAM2 weights path")
    p.add_argument("--frames-path", required=True, 
                   help="Directory containing frame images for the current participant")
    p.add_argument("--out-path", required=True, 
                   help="Directory where output overlays and masks will be stored")
    p.add_argument("--json-path",
                   default="/mnt/data/wetcat_dataset/wetcat_code/sam2_project/img413.json",
                   help="Path to annotations JSON ")
    return p.parse_args()

def main():
    args = parse_args()

    CHECKPOINT = os.path.abspath(args.checkpoint_path)
    FRAME_DIR = os.path.abspath(args.frames_path)
    BASE_OUTPUT_DIR = os.path.abspath(args.out_path)
    JSON_PATH = os.path.abspath(args.json_path)
    MODEL_CFG = "sam2_hiera_l.yaml"

    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    DIRS = {
        "overlays": os.path.join(BASE_OUTPUT_DIR, "overlays"),
        "masks": os.path.join(BASE_OUTPUT_DIR, "masks")
    }
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

    # 1: Green (Left), 2: Blue (Right), 3: Magenta (Needle), 4: Yellow (Iris)
    colors = {1: [0, 255, 0], 2: [255, 0, 0], 3: [255, 0, 255], 4: [0, 255, 255]}

    all_files = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for i, filename in enumerate(all_files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i:05d}{ext}"
        if filename != new_name:
            os.rename(os.path.join(FRAME_DIR, filename), os.path.join(FRAME_DIR, new_name))
    
    frame_names = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device)
    inference_state = predictor.init_state(video_path=FRAME_DIR)
    predictor.reset_state(inference_state)

    ANCHOR_FRAME_IDX = 0 
    sample_frame = cv2.imread(os.path.join(FRAME_DIR, frame_names[ANCHOR_FRAME_IDX]))
    h, w = sample_frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    pince_shapes = sorted([s for s in data['shapes'] if s['label'] == 'Pince'], 
                          key=lambda s: np.mean(np.array(s['points'])[:, 0]))

    best_needle_anchor = None
    min_dist_to_center = float('inf')


    for shape in data['shapes']:
        if shape['label'] == 'Iris':
            pts = np.array(shape['points'], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.bool_)
            temp = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp, [pts], 1)
            mask[temp == 1] = True
            predictor.add_new_mask(inference_state, ANCHOR_FRAME_IDX, 4, mask)

    for i, shape in enumerate(pince_shapes):
        obj_id = i + 1 
        pts = np.array(shape['points'], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.bool_)
        temp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(temp, [pts], 1)
        mask[temp == 1] = True
        predictor.add_new_mask(inference_state, ANCHOR_FRAME_IDX, obj_id, mask)
        
        dists = np.sqrt((pts[:, 0] - center_x)**2 + (pts[:, 1] - center_y)**2)
        tip_idx = np.argmin(dists)
        if dists[tip_idx] < min_dist_to_center:
            min_dist_to_center = dists[tip_idx]
            best_needle_anchor = pts[tip_idx]

    if best_needle_anchor is not None:
        ax, ay = best_needle_anchor
        needle_pts = np.array([
            [ax - 20, ay],      
            [ax - 50, ay - 5],  
            [ax - 80, ay - 10], 
            [ax - 40, ay + 40], 
            [ax - 40, ay - 40]  
        ], dtype=np.float32)
        needle_labels = np.array([1, 1, 1, 0, 0], dtype=np.int32) 
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ANCHOR_FRAME_IDX,
            obj_id=3,
            points=needle_pts,
            labels=needle_labels
        )

    video_segments = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = (out_obj_ids, out_mask_logits)

    print("Saving processed frames:")
    for out_frame_idx in sorted(video_segments.keys()):
        out_obj_ids, out_mask_logits = video_segments[out_frame_idx]
        img_name = frame_names[out_frame_idx]
        
        frame = cv2.imread(os.path.join(FRAME_DIR, img_name))
        overlay, mask_canvas = frame.copy(), np.zeros_like(frame)

        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            if mask.any():
                color = colors.get(obj_id, [255, 255, 255])
                overlay[mask] = color
                mask_canvas[mask] = color

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, overlay)
        cv2.imwrite(os.path.join(DIRS["overlays"], f"overlay_{img_name}"), overlay)
        cv2.imwrite(os.path.join(DIRS["masks"], f"mask_{img_name}"), mask_canvas)


if __name__ == "__main__":
    main()
