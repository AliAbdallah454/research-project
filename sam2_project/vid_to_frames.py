#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:05:42 2026

@author: djayadeep
"""
# "Convert video to frames"
import os
import subprocess

video_path = '/mnt/data/wetcat_dataset/wetcat_code/test-video/gepromed.mp4'
frame_dir = '/mnt/data/wetcat_dataset/wetcat_code/sam2_project/frames_sam/'

os.makedirs(frame_dir, exist_ok=True)


command = [
    'ffmpeg', 
    '-i', video_path, 
    '-q:v', '2', 
    f'{frame_dir}/%05d.jpg'
]


try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"{e}")
