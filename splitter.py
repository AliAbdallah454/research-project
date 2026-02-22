# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan 22 14:26:28 2026

# @author: djayadeep
# """

# import pandas as pd
# import cv2
# import os

# # --- CONFIGURATION ---
# # Path to the folder containing your LONG videos (e.g., wetlab_cataract_005.mp4)
# raw_videos_dir = 'Data/' 
# # Path to your CSV file
# csv_path = 'Data/wetlab_cataract_005_phases.csv' 
# # Where the splits will be saved
# base_output_dir = 'sData/plits/'

# # 1. Load the annotations
# df = pd.read_csv(csv_path)

# # 2. Get the Video ID (assuming it matches the filename)
# # Adjust this if your CSV has a column for video_id
# video_id = "wetlab_cataract_005" 
# video_filename = f"{video_id}.mp4"
# full_video_path = os.path.join(raw_videos_dir, video_filename)

# # 3. Create the specific folder for this video session
# session_dir = os.path.join(base_output_dir, video_id)
# if not os.path.exists(session_dir):
#     os.makedirs(session_dir)

# # 4. Initialize counters for naming (e.g., idle_clip_1, idle_clip_2)
# phase_counters = {}

# # 5. Process the video
# cap = cv2.VideoCapture(full_video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# for index, row in df.iterrows():
#     phase_name = row['phase'].lower().strip() # e.g., "idle"
#     start_frame = int(row['start'])
#     end_frame = int(row['end'])
    
#     # Update counter for this specific phase
#     phase_counters[phase_name] = phase_counters.get(phase_name, 0) + 1
#     count = phase_counters[phase_name]
    
#     # Construct filename: splits/wetlab_cataract_005/idle_clip_1.mp4
#     clip_name = f"{phase_name}_clip_{count}.mp4"
#     output_path = os.path.join(session_dir, clip_name)
    
#     # Extract frames
#     print(f"Exporting {clip_name} (Frames {start_frame} to {end_frame})...")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     for f in range(start_frame, end_frame):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
    
#     out.release()

# cap.release()
# print("Done! Your 'splits/' folder is ready for training.")

import pandas as pd
import cv2
import os

# --- CONFIGURATION ---
raw_videos_dir = 'Data/' 
csv_path = 'phase_csvs_Dataset_Paper/wetlab_cataract_009_phases.csv' 
base_output_dir = 'sData/plits1/'

# 1. Load and clean annotations
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]

# 2. Get the Video ID
video_id = "wetlab_cataract_009" 
full_video_path = os.path.join(raw_videos_dir, f"{video_id}.mp4")

session_dir = os.path.join(base_output_dir, video_id)
os.makedirs(session_dir, exist_ok=True)

# 3. Open Video and Get Properties
cap = cv2.VideoCapture(full_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

phase_counters = {}

for index, row in df.iterrows():
    phase_name = str(row['phase']).lower().strip()
    
    # --- DECIMAL TO FRAME CONVERSION ---
    # If your CSV has seconds (e.g. 12.5), we multiply by FPS
    start_frame = int(float(row['start']) * fps)
    end_frame = int(float(row['end']) * fps)
    
    # SAFETY: Ensure we don't go past the end of the video
    if end_frame > total_frames:
        end_frame = total_frames
    
    phase_counters[phase_name] = phase_counters.get(phase_name, 0) + 1
    count = phase_counters[phase_name]
    
    clip_name = f"{phase_name}_clip_{count}.mp4"
    output_path = os.path.join(session_dir, clip_name)
    
    print(f"Exporting {clip_name}: {row['start']}s to {row['end']}s (Frames {start_frame}-{end_frame})")
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    out.release() # This "closes" the file properly

cap.release()
print("Process Complete.")