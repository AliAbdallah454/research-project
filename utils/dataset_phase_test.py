import os
import csv
import random
import torch
import torchvision.transforms as transforms
import torchvision.io as tvio
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicDataset(Dataset):
    def __init__(self, csv_file, batch_size=16, num_frames=10, clip_duration=90, size=(256, 256), is_train=True):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.size = size
        self.is_train = is_train
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.data = list(reader)
        for row in self.data:
            if row[1] in ["idle", "rest"]:
                row[1] = "idle_rest"

        # Step 2: Get unique phases and class count
        self.phases = list(set(row[1] for row in self.data))
        self.phases.sort()
        self.num_classes = len(self.phases)
        
        print(f'num classes: {self.num_classes}')
        self.video_list = []
        for row in self.data:
            video_name, phase, video_path, start_time = row
            self.video_list.append((video_name, phase, video_path, int(start_time)))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
        ])
    
    def __len__(self):
        return len(self.video_list)
    
    def load_video_clip(self, video_path, start_time, phase):
        cap = cv2.VideoCapture(video_path)
        frame_step = self.clip_duration // self.num_frames

        frames = []
        frame_labels = []
        
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30 + i * frame_step) 
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = self.transform(frame)  # Apply transformations
            frames.append(frame)
            frame_labels.append(torch.tensor(int(self.phases.index(phase))))
        cap.release()
        
        if len(frames) == self.num_frames:
            return torch.stack(frames), torch.tensor(frame_labels)  # Frames * Channels * Width * Height
        return None
        
    
    def __getitem__(self, idx):
        video_name, phase, video_path, start_time = self.video_list[idx]
        result = self.load_video_clip(video_path, int(start_time), phase)
        if result is None:
                # Option 1: skip or retry (for training)
                return self.__getitem__(idx)
        video_clip, video_label = result
        return {'video': video_clip.float().to(device), 'label': video_label.long().to(device)}

if __name__ == "__main__":
    split_dir = "/storage/workspaces/artorg_aimi/ws_00000/Negin/Cataract_Else/Wet_Lab/splits/"
    output_dir = "/storage/homefs/ng22l920/Codes/Wetlab_Phase/TrainIDs/"
    train_csv = os.path.join(output_dir, "test_fold_1_sequences.csv")

    dataset = BasicDataset(train_csv, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader:
        print(batch['video'].shape, batch['label'].shape)
        # break
