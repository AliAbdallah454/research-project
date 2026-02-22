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
    def __init__(self, csv_file, batch_size=15, num_frames=10, clip_duration=90, size=(256, 256)):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.size = size

        if 'train' in self.csv_file:
            self.is_train = True
        else:
             self.is_train = False   
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.data = list(reader)
        
        self.videos = self.filter_short_videos(self.data)
        # self.phases = list(set(row[1] for row in self.data))
        # self.phases.sort()
        # self.num_classes = len(self.phases)
        # print(f'num classes: {self.num_classes}')

        # Step 1: Normalize labels in self.data
        for row in self.data:
            if row[1] in ["idle", "rest"]:
                row[1] = "idle_rest"

        # Step 2: Get unique phases and class count
        self.phases = list(set(row[1] for row in self.data))
        self.phases.sort()
        self.num_classes = len(self.phases)

        print(f'num classes: {self.num_classes}')
        print(f'phases: {self.phases}')
        self.videos_per_class = self.batch_size // self.num_classes
        
        self.phase_videos = []
        for phase in self.phases:
            self.phase_videos.append([row for row in self.videos if row[1] == phase])
        # print(f'self.phase_videos: {self.phase_videos}')
        # Define transformations
        if self.is_train:
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # Zoom in and zoom out (scales between 80% to 120% of the original size)
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), ratio=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomGrayscale(p=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def filter_short_videos(self, video_list):
        filtered_list = []
        for entry in video_list:
            video_path = entry[2]
            if "AI_video_010" in video_path:
                print(f"Removing {video_path} (Excluded by name filter)")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Skipping...")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count >= 90:
                filtered_list.append(entry)
            else:
                print(f"Removing {video_path} (Only {frame_count} frames)")

        return filtered_list

    def __len__(self):
        return len(self.data)//self.batch_size
    
    def load_video_clip(self, video_path, phase):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f'total_frames: {total_frames}')
        if total_frames < self.clip_duration:
            cap.release()
            return None
        
        start_frame = random.randint(0, total_frames - self.clip_duration)
        frame_indices = sorted(random.sample(range(start_frame, start_frame+self.clip_duration), self.num_frames))

        frames = []
        frame_labels = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.resize(frame, self.size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = self.transform(frame)  # Apply transformations
            frames.append(frame)
            frame_labels.append(torch.tensor(int(self.phases.index(phase))))
        cap.release()
        
        if len(frames) == self.num_frames:
            return torch.stack(frames), torch.tensor(frame_labels), video_path  # Frames * Channels * Width * Height
        return None
    
    def __getitem__(self, idx):
        selected_videos = []
        for i in range(self.num_classes):
            available_videos = random.sample(self.phase_videos[i], self.videos_per_class)
            selected_videos.append(available_videos)
        flattened_videos = list(itertools.chain.from_iterable(selected_videos))

        vids = []
        labels = []
        paths = []
        
        for selected_video in flattened_videos:
            _, phase, video_path = selected_video
            result = self.load_video_clip(video_path, phase)
            if result is None:
                # Option 1: skip or retry (for training)
                return self.__getitem__(idx)

                # Option 2: raise error (for debugging)
                # raise ValueError(f"Failed to load video clip for: {video_path}, phase: {phase}")

            video_clip, video_label, video_path = result
            if video_clip is not None:
                vids.append(video_clip)
                labels.append(video_label)
                paths.append(video_path)
        # print(paths)
        return {'video': torch.stack(vids).float().to(device), 'label': torch.stack(labels).long().to(device)}

if __name__ == "__main__":
    split_dir = "/storage/workspaces/artorg_aimi/ws_00000/Negin/Cataract_Else/Wet_Lab/splits/"
    output_dir = "/storage/homefs/ng22l920/Codes/Wetlab/TrainIDs_Phase/"
    train_csv = os.path.join(output_dir, "train_fold_1.csv")

    dataset = BasicDataset(train_csv, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch['video'].shape, batch['label'].shape)
        print(batch['label'])
        break
