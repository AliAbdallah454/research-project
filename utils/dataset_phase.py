import os
import csv
import random
import torch
import torchvision.io as tvio
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicDataset(Dataset):
    def __init__(self, csv_file, batch_size=16, num_frames=10, clip_duration=3, size=(256, 256)):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.size = size
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.data = list(reader)

        
        
        self.videos = self.filter_short_videos(self.data)
        self.phases = list(set(row[1] for row in self.data))
        self.phases.sort()
        self.num_classes = len(self.phases)
        print(f'num classes: {self.num_classes}')
        self.videos_per_class = self.batch_size // self.num_classes
        # print(f'data: {self.data}')

        self.phase_videos = []
        for phase in self.phases:
            self.phase_videos.append([row for row in self.videos if row[1] == phase])


    def filter_short_videos(self, video_list):
        """
        Reads videos from the given list and removes:
        - Entries where the video has fewer than 75 frames.
        - Entries where the video path contains 'AI_video_010'.

        Parameters:
            video_list (list): A list of lists, where each inner list contains [name, phase, video_path].

        Returns:
            list: A filtered list containing only valid videos.
        """
        filtered_list = []

        for entry in video_list:
            video_path = entry[2]  # Extract video path
            

            # Remove videos with "AI_video_010" in the path
            if "AI_video_010" in video_path:
                print(f"Removing {video_path} (Excluded by name filter)")
                continue

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Skipping...")
                continue

            # Get frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()  # Close the video file

            # Check frame length condition
            if frame_count >= 75:
                filtered_list.append(entry)
            else:
                print(f"Removing {video_path} (Only {frame_count} frames)")

        return filtered_list

    def __len__(self):
        return len(self.data) // self.batch_size
    
    def load_video_clip(self, video_path, phase):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.clip_duration * self.num_frames:
            print('error 1')
            cap.release()
            return None
        
        start_frame = random.randint(0, total_frames - self.clip_duration)
        frame_step = self.clip_duration // self.num_frames
        frames = []
        frame_labels = []
        
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * frame_step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.size)
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)
            frame_labels.append(torch.tensor(int(self.phases.index(phase))))
        cap.release()
        
        if len(frames) == self.num_frames:
            return torch.tensor(np.array(frames)).permute(0, 3, 1, 2), torch.tensor(frame_labels)  # Frames * Channels * Width * Height
        print('error 2')
        return None
    
    def __getitem__(self, idx):
        
        
        selected_videos = []
        
       
        for i in range(self.num_classes):
            
            available_videos = random.sample(self.phase_videos[i], self.videos_per_class)
            selected_videos.append(available_videos)
        flattened_videos = list(itertools.chain.from_iterable(selected_videos))    
        # print(f'selected_videos: {selected_videos}')

        vids = []
        labels = []
        
        
        for selected_video in flattened_videos:
            print(selected_video)
            _, phase, video_path=selected_video
            # print(f'phase:{phase}')
            # print(f'video_path:{video_path}')
            video_clip, video_label = self.load_video_clip(video_path, phase)
            if video_clip is not None:
                vids.append(video_clip)
                labels.append(video_label)
                    
        
        return {'video': torch.stack(vids).float().to(device), 'label': torch.stack(labels).long().to(device)}

if __name__ == "__main__":
    split_dir = "/storage/workspaces/artorg_aimi/ws_00000/Negin/Cataract_Else/Wet_Lab/splits/"
    output_dir = "/storage/homefs/ng22l920/Codes/Wetlab_Phase/TrainIDs/"
    train_csv = os.path.join(output_dir, "train_fold_1.csv")

    dataset = BasicDataset(train_csv, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch['video'].shape, batch['label'].shape)
        break
