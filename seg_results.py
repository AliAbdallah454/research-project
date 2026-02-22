#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:36:32 2026

@author: djayadeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import numpy as np
import importlib
import time
from torchvision import transforms 

sys.path.append(os.getcwd())

from utils.import_helper import import_config

def main():
    config_file = 'configs_Seg.AnatomyInst.Config_Supervised_DeepLabV3_Res50'
    my_conf = importlib.import_module(config_file)
    
    params = import_config.execute(my_conf)
    num_classes = 1  
    Results_path = params[9]
    Checkpoint_path = params[17]
    Framework_name = params[3]
    dataset_name = os.path.basename(params[2][0][0]) 
    net_name = params[15]
    batch_size = params[7]
    GCC = my_conf.GCC 
    SemiSupervised_initial_epoch = params[21]
    Learning_Rate = params[5][0]
    affine = params[23]
    Net1 = params[18]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net = Net1('resnet50', num_classes)
    
    dir_checkpoint = f"{Results_path}{Checkpoint_path}{Framework_name}_{dataset_name}_{net_name}_BS_{batch_size}_GCC_{GCC}_init_epoch_{SemiSupervised_initial_epoch}_{Learning_Rate}_Affine_{affine}/"
    load_path = os.path.join(dir_checkpoint, 'CP_epoch80.pth')
    
    if not os.path.exists(load_path):
        print(f"ERROR: Could not find checkpoint at {load_path}")
        return

    print(f"Loading weights from: {load_path}")
    state_dict = torch.load(load_path, map_location=device)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    video_path = '/mnt/data/wetcat_dataset/wetcat_code/test-video/wetlab_cataract_001_1.mp4'
    output_path = '/mnt/data/wetcat_dataset/wetcat_code/output/masks/'
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Starting Inference... Looking for Class 1 (Instruments)")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % 3 == 0:
            start_time = time.time()

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = inference_transform(img_rgb).to(device).unsqueeze(0)

            with torch.no_grad():
                output = net(input_tensor)
            
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            binary_mask = np.zeros_like(mask)
            binary_mask[mask == 1] = 255 
            
            final_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            mask_filename = os.path.join(output_path, f"mask_{frame_idx:04d}.png")
            cv2.imwrite(mask_filename, final_mask)

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                unique = np.unique(mask)
                print(f"Frame {frame_idx}: Found Classes {unique} ({elapsed:.3f}s)")

        frame_idx += 1

    cap.release()
    print(f"masks saved to: {output_path}")

if __name__ == '__main__':
    main()
