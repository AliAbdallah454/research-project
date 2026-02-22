#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:46:23 2026

@author: djayadeep
"""

import os
import csv
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, csv_file, batch_size=1, size=(256, 256), is_train=True, *args, **kwargs):
        self.csv_file = csv_file
        self.size = size
        self.is_train = is_train
        self.batch_size = batch_size

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            # CRITICAL: Skip the header row ("Unnamed: 0,imgs,masks")
            header = next(reader) 
            self.data = list(reader)

        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Handle index errors safely
        if idx >= len(self.data):
            return self.__getitem__(0)
            
        row = self.data[idx]
        
        # CORRECTED INDICES for your specific CSV format:
        # row[0] is the ID number (ignore)
        # row[1] is the Image Path
        # row[2] is the Mask Path
        img_path = row[1]  
        mask_path = row[2] 

        # 1. Load Image
        # Check if path is absolute or relative. If relative, you might need to prepend a root dir.
        # Assuming paths in CSV (e.g., "frames/...") are correct relative to where you run the script.
        image = cv2.imread(img_path)
        if image is None:
            # Try prepending a root directory if needed, or just skip
            # print(f"Warning: Could not load image {img_path} - Retrying with next sample")
            return self.__getitem__((idx + 1) % len(self.data))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load Mask (GrayScale)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            # print(f"Warning: Could not load mask {mask_path}")
            return self.__getitem__((idx + 1) % len(self.data))

        # 3. Resize Mask (Nearest Neighbor to keep class integers)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to Tensor (Long for CrossEntropy)
        # Unique values in mask should be 0, 1, 2...
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        # 4. Transform Image
        img_tensor = self.img_transform(image)

        return {
            'video': img_tensor,  # Name kept as 'video' to match your training loop
            'label': mask_tensor,
            'path': img_path
        }