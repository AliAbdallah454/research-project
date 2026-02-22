#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  19 20:43:09 2026

@author: djayadeep
"""

import os
import sys
import cv2
import torch
import numpy as np
import importlib
import time
from torchvision.transforms import Resize, ToTensor

# Adding current directory to sys.path to ensure custom imports work
sys.path.append(os.getcwd())

# Custom Imports from the WetCat repo
from utils.import_helper import import_config

def main():
    # 1. SETUP CONFIGS
    config_file = 'configs_Seg.AnatomyInst.Config_Supervised_DeepLabV3_Res50'
    my_conf = importlib.import_module(config_file)
    
    # Extracting only what we need from the config helper
    # We need Net1 (the class), num_classes, and path info
    params = import_config.execute(my_conf)
    num_classes = 2
    Results_path = params[9]
    Checkpoint_path = params[17]
    Framework_name = params[3]
    dataset_name = params[2][0][0]
    net_name = params[15]
    batch_size = params[7]
    GCC = my_conf.GCC
    SemiSupervised_initial_epoch = params[21]
    Learning_Rate = params[5][0]
    affine = params[23]
    Net1 = params[18]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. INITIALIZE MODEL
    # Standard ResNet50 DeepLabV3+ call based on the repo structure
    net = Net1('resnet50', num_classes)
    
    # 3. LOAD YOUR TRAINED WEIGHTS (Checkpoint 16)
    dir_checkpoint = f"{Results_path}{Checkpoint_path}{Framework_name}_{dataset_name}_{net_name}_BS_{batch_size}_GCC_{GCC}_init_epoch_{SemiSupervised_initial_epoch}_{Learning_Rate}_Affine_{affine}/"
    load_path = os.path.join(dir_checkpoint, 'CP_epoch80.pth')
    
    if not os.path.exists(load_path):
        print(f"ERROR: Could not find checkpoint at {load_path}")
        return

    print(f"Loading weights from: {load_path}")
    state_dict = torch.load(load_path, map_location=device)
    
    # Handle the 'module.' prefix if it exists from DataParallel training
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # 4. VIDEO PROCESSING SETUP
    video_path = '/mnt/data/wetcat_dataset/wetcat_code/test-video/gepromed.mp4'
    output_path = '/mnt/data/wetcat_dataset/wetcat_code/output/masks/gepromed/'
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    resize_transform = Resize((512, 512))
    to_tensor = ToTensor()

    print("Starting Inference...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 3rd frame to save time (approx 10fps for a 30fps video)
        if frame_idx % 3 == 0:
            start_time = time.time()

            # Prepare Input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = to_tensor(frame_rgb).to(device)
            input_tensor = resize_transform(input_tensor).unsqueeze(0) # [1, 3, 512, 512]

            # Run Inference
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output = net(input_tensor) # [1, num_classes, 512, 512]
            
            # Extract Mask
            # Get class with highest probability for each pixel
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Resize back to original video dimensions
            mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            # Save the raw mask (multiply by 60 to make classes 0, 60, 120, 180 visible)
            mask_filename = os.path.join(output_path, f"mask_{frame_idx:04d}.png")
            cv2.imwrite(mask_filename, mask_resized * 60)

            elapsed = time.time() - start_time
            print(f"Processed frame {frame_idx} ({elapsed:.3f}s)")

        frame_idx += 1

    cap.release()
    print(f"Done! Masks saved to: {output_path}")

if __name__ == '__main__':
    main()
