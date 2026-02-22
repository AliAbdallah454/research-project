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
        if idx >= len(self.data):
            return self.__getitem__(0)
            
        row = self.data[idx]
        img_path = row[1]  
        mask_path = row[2] 

        # 1. Load Image
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self.data))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load Mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            return self.__getitem__((idx + 1) % len(self.data))

        # 3. Resize Mask
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        mask[mask > 0] = 1 
    
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        img_tensor = self.img_transform(image)

        return {
            'video': img_tensor,
            'label': mask_tensor,
            'path': img_path
        }
