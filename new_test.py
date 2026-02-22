#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:40:48 2026

@author: djayadeep
"""


import argparse
import logging
import os
import sys
import cv2
import torch
import numpy as np
from torchvision.transforms import Resize, ToTensor
from torchvision.io import write_video
import torch.nn as nn

# Custom Imports

from utils.import_helper import import_config
from scipy.ndimage import binary_fill_holes
import av
import time
from torch.cuda.amp import autocast
from skimage.morphology import skeletonize



# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
import importlib


class Printer(nn.Module):
    def __init__(self, global_dict=globals()):
        super(Printer, self).__init__()
        self.global_dict = global_dict
        self.except_list = []

    def debug(self, expression):
        frame = sys._getframe(1)
        print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))

    def namestr(self, obj, namespace):
        return [name for name in namespace if namespace[name] is obj]     

    def forward(self, x):
        for i in x:
            if i not in self.except_list:
                name = self.namestr(i, globals())
                if len(name) > 1:
                    self.except_list.append(i)
                    for j in range(len(name)):
                        self.debug(name[j])
                else:  
                    self.debug(name[0])

def refine_instrument_mask(mask, min_area=100):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create an empty mask for refined components
    refined_mask = np.zeros_like(mask)

    # Iterate through connected components
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:  # Filter small components
            refined_mask[labels == label] = 1

    # Apply morphological operations (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

    return refined_mask

def process_video_frames(net, video_path, output_video_path, device, resize_transform, desired_fps=10, max_duration=-1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = int(original_fps / desired_fps)

    # Get the input video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Determine maximum number of frames to process
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames if max_duration == -1 else int(max_duration * original_fps)

    # Define codecs and create VideoWriter objects
    output_video_no_morph = os.path.join(output_video_path, f"{video_name}_no_morph.mp4")
    output_video_morph = os.path.join(output_video_path, f"{video_name}_morph.mp4")
    output_video_ellipses = os.path.join(output_video_path, f"{video_name}_ellipses.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out_no_morph = cv2.VideoWriter(output_video_no_morph, fourcc, desired_fps, (frame_width, frame_height))
    out_morph = cv2.VideoWriter(output_video_morph, fourcc, desired_fps, (frame_width, frame_height))
    out_ellipses = cv2.VideoWriter(output_video_ellipses, fourcc, desired_fps, (frame_width, frame_height))

    frame_idx = 0
    max_time_per_frame = 1.0 / desired_fps  # Maximum time available per frame for real-time processing
    total_processing_time = 0.0  # Track total processing time for all frames
    num_processed_frames = 0    # Track the number of processed frames
    frame_buffer = []
    while cap.isOpened():
        if max_duration != -1 and frame_idx >= max_frames:  # Stop processing after max_duration frames unless set to -1
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            start_time = time.time()  # Start timing

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_tensor = ToTensor()(frame_rgb).unsqueeze(0).unsqueeze(0).to(device)  # Add batch dim
            # frame_tensor = resize_transform(frame_tensor)
            frame_tensor = ToTensor()(frame_rgb).to(device)
            frame_tensor = resize_transform(frame_tensor)



            start_time1 = time.time()
            frame_sequence = frame_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 10, 1, 1, 1)
            

            with autocast(): # Reducing inference time around 0.1s per frame!
                 mask_pred = net(frame_sequence)

            current_frame_logits = mask_pred[0, -1]
            final_mask = torch.argmax(mask_pred, dim=1).cpu().numpy().astype(np.uint8)[0]
            final_mask_value = torch.argmax(current_frame_logits).item()

            final_mask = np.full((frame_height, frame_width), final_mask_value, dtype=np.uint8)


            end_time1 = time.time()  # End timing
            frame_processing_time1 = end_time1 - start_time1

            print(f"Frame {frame_idx}: Prediction Time = {frame_processing_time1:.3f}s")
            if frame_processing_time1 > max_time_per_frame:
                print(f"WARNING: Frame {frame_idx} prediction time exceeds real-time requirement ({frame_processing_time1:.3f}s > {max_time_per_frame:.3f}s)")

            # Resize the final_mask to match the frame dimensions
            final_mask_resized = cv2.resize(final_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            # Without morphological operations (colored overlay)
            overlay_no_morph = frame.copy()
            colored_mask_no_morph = cv2.applyColorMap(final_mask_resized * 40, cv2.COLORMAP_JET)  # Example scaling
            overlay_no_morph = cv2.addWeighted(overlay_no_morph, 0.6, colored_mask_no_morph, 0.4, 0)

            # With morphological operations
            final_mask_morph = cv2.morphologyEx(final_mask_resized, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            overlay_morph = frame.copy()
            colored_mask_morph = cv2.applyColorMap(final_mask_morph * 40, cv2.COLORMAP_JET)
            overlay_morph = cv2.addWeighted(overlay_morph, 0.6, colored_mask_morph, 0.4, 0)

            # With ellipses
            overlay_ellipses = frame.copy()
            # Initialize variables to store pupil and lens areas
            pupil_area = None
            lens_area = None

            # Divide the frame into 3x3 patches
            frame_height, frame_width = frame.shape[:2]
            patch_height = frame_height // 7
            patch_width = frame_width // 7

            # Central patch coordinates
            central_patch_x_min = patch_width
            central_patch_x_max = 6 * patch_width
            central_patch_y_min = patch_height
            central_patch_y_max = 6 * patch_height

            for class_id in [1, 2, 3]:  # Process iris, pupil, and instrument
                if class_id == 1:  # Combine class_id 1 (Iris) and class_id 2 (Pupil)
                    binary_mask = ((final_mask_morph == 1) | (final_mask_morph == 2)).astype(np.uint8)
                else:  # For other class IDs, keep their masks separate
                    binary_mask = (final_mask_morph == class_id).astype(np.uint8)
                
                binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)  # Fill holes in the mask
                
                if class_id == 1:  # Special handling for the Iris
                    # Perform connected component analysis
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                    
                    # Combine all components into a single mask
                    merged_pupil_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                    for label in range(1, num_labels):  # Ignore background (label 0)
                        merged_pupil_mask[labels == label] = 1
                    
                    # Update the binary mask for ellipse fitting
                    binary_mask = merged_pupil_mask
                
                # Find contours for the (possibly updated) binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Store pupil contour for later use with instruments
                    if class_id == 2:
                        pupil_contour = largest_contour
                        pupil_mask = binary_mask.copy()  # Store pupil mask for later use
                    
                    if len(largest_contour) >= 5:  # Ellipse fitting requires at least 5 points
                        ellipse = cv2.fitEllipse(largest_contour)
                        area = cv2.contourArea(largest_contour)  # Calculate the area of the contour
                        center_x, center_y = ellipse[0]  # Ellipse center coordinates
                        
                        # Check if the ellipse center is in the central patch
                        if central_patch_x_min <= center_x <= central_patch_x_max and central_patch_y_min <= center_y <= central_patch_y_max:
                            # Handle different classes
                            if class_id == 1:  # Iris
                                Iris_area = area
                                cv2.ellipse(overlay_ellipses, ellipse, (0, 255, 0), 2)  # Green for pupil
                            elif class_id == 2:  # Pupil
                                Pupil_area = area
                                if Iris_area is not None and Iris_area >= Pupil_area / 3:  # Validate lens area
                                    cv2.ellipse(overlay_ellipses, ellipse, (255, 0, 0), 2)  # Blue for lens
                            # else:  # For class_id == 3 (instrument) or other classes
                            #     cv2.ellipse(overlay_ellipses, ellipse, (0, 255, 255), 2)  # Yellow for other classes
                        else:
                            # If the center is not in the central patch, overlay text
                            cv2.putText(overlay_ellipses, "Eye not centered", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (0, 0, 255), 2, cv2.LINE_AA)
            instrument_mask = (final_mask_morph == 3).astype(np.uint8)
            instrument_mask = refine_instrument_mask(instrument_mask)
            instrument_contours, _ = cv2.findContours(instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours based on their topmost y-coordinate (ascending order)
            sorted_contours = sorted(instrument_contours, key=lambda c: np.min(c[:, :, 1]))

            # Process the two highest contours
            # for contour in sorted_contours[:2]:
            #     if len(contour) > 0:
            #         # Get bounding box for the instrument
            #         x, y, w, h = cv2.boundingRect(contour)
                    
            #         # Determine instrument tip locations (top-left and top-right corners)
            #         top_left = (x, y)
            #         top_right = (x + w, y)
                    
            #         # Check if both tips are inside the pupil mask
            #         if 'pupil_mask' in locals():
            #             top_left_in_pupil = (
            #                 0 <= top_left[0] < pupil_mask.shape[1] and 
            #                 0 <= top_left[1] < pupil_mask.shape[0] and 
            #                 pupil_mask[top_left[1], top_left[0]] > 0
            #             )
            #             top_right_in_pupil = (
            #                 0 <= top_right[0] < pupil_mask.shape[1] and 
            #                 0 <= top_right[1] < pupil_mask.shape[0] and 
            #                 pupil_mask[top_right[1], top_right[0]] > 0
            #             )
                        
            #             # If both tips are inside the pupil, draw circles at their positions
            #             if top_left_in_pupil and top_right_in_pupil:
            #                 cv2.circle(overlay_ellipses, top_left, radius=5, color=(0, 165, 255), thickness=-1)  # Orange circle for top-left tip
            #                 cv2.circle(overlay_ellipses, top_right, radius=5, color=(0, 165, 255), thickness=-1)  # Orange circle for top-right tip
            # Handle instruments (class_id = 3) separately to check if tip is in pupil
            # instrument_mask = (final_mask_morph == 3).astype(np.uint8)
            # instrument_mask = refine_instrument_mask(instrument_mask)
            # instrument_contours, _ = cv2.findContours(instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Process each instrument contour
            # for contour in instrument_contours:
            #     if len(contour) > 0:
            #         # Get bounding box for the instrument
            #         x, y, w, h = cv2.boundingRect(contour)
                    
            #         # Determine instrument tip locations (top-left and top-right corners)
            #         top_left = (x, y)
            #         top_right = (x + w, y)
                    
            #         # Check if either tip is inside the pupil mask
            #         if 'pupil_mask' in locals():
            #             # Check if top-left tip is inside pupil
            #             if 0 <= top_left[0] < pupil_mask.shape[1] and 0 <= top_left[1] < pupil_mask.shape[0]:
            #                 top_left_in_pupil = pupil_mask[top_left[1], top_left[0]] > 0
            #             else:
            #                 top_left_in_pupil = False
                            
            #             # Check if top-right tip is inside pupil
            #             if 0 <= top_right[0] < pupil_mask.shape[1] and 0 <= top_right[1] < pupil_mask.shape[0]:
            #                 top_right_in_pupil = pupil_mask[top_right[1], top_right[0]] > 0
            #             else:
            #                 top_right_in_pupil = False
                        
            #             # If either tip is inside the pupil, draw the bounding box
            #             if top_left_in_pupil or top_right_in_pupil:
            #                 cv2.rectangle(overlay_ellipses, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Orange for instrument

            end_time = time.time()  # End timing
            frame_processing_time = end_time - start_time
            total_processing_time += frame_processing_time
            num_processed_frames += 1

            # Print processing time for the current frame
            print(f"Frame {frame_idx}: Processing Time = {frame_processing_time:.3f}s")
            if frame_processing_time > max_time_per_frame:
                print(f"WARNING: Frame {frame_idx} processing exceeds real-time requirement ({frame_processing_time:.3f}s > {max_time_per_frame:.3f}s)")


              
            # Write frames to respective video files
            out_no_morph.write(overlay_no_morph)
            out_morph.write(overlay_morph)
            out_ellipses.write(overlay_ellipses)

        frame_idx += 1

    cap.release()
    out_no_morph.release()
    out_morph.release()
    out_ellipses.release()
    avg_processing_time = total_processing_time / num_processed_frames if num_processed_frames > 0 else 0
    print(f"Average Processing Time per Frame: {avg_processing_time:.3f}s")
    print(f"Real-Time Requirement Met: {avg_processing_time <= max_time_per_frame}")

    print(f"Videos saved as:")
    print(f" - {output_video_no_morph}")
    print(f" - {output_video_morph}")
    print(f" - {output_video_ellipses}")







def train_net(net, epochs, batch_size, lr, device, video_path, output_video_path):
    resize_transform = Resize((512, 512))
    process_video_frames(net, video_path, output_video_path, device, resize_transform)


if __name__ == '__main__':
    
    config_file = 'configs_Seg.AnatomyInst.Config_Supervised_DeepLabV3_Res50'
    # config_file = 'configs.Config'
    # config_file = 'configs_Seg.rhexis.Config_Supervised_DeepLabV3_Res50'
    
    my_conf = importlib.import_module(config_file)
    
    criterion_supervised, criterion_SemiSupervised, datasets, Framework_name, num_classes, \
             Learning_Rates_init, epochs, batch_size, size,\
                 Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, SemiSupervised_batch_size, SemiSupervised_initial_epoch,\
                             image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, TrainIDs_path\
                     = import_config.execute(my_conf)

    print("inside main")
    print('Hello Ubelix')
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    #print(f'Cuda Device Number: {torch.cuda.current_device()}')
    #print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    printer1 = Printer()
    print('CONFIGS:________________________________________________________')
    printer1([criterion_supervised, criterion_SemiSupervised, datasets, Framework_name, num_classes, \
             Learning_Rates_init, epochs, batch_size, size,\
                 Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, SemiSupervised_batch_size, SemiSupervised_initial_epoch,\
                             image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, TrainIDs_path])

    video_path = '/content/drive/MyDrive/WetCat-main/wetlab_cataract_001_1.mp4'
    output_video_path = '/content/drive/MyDrive/WetCat-main/output_vids/'
    desired_fps = 10

    try:
        for c in range(1):      
            for LR in range(len(Learning_Rates_init)):
                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')
                
                

                # net = Net1('resnet50', num_classes)
                net = Net1
                
                net = torch.nn.parallel.DataParallel(net, device_ids=list(range(num_gpus)))
                

                net.to(device=device)

                net.eval()

                # net = torch.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)

                dataset_name = datasets[c][0]
                dir_checkpoint = Results_path + Checkpoint_path +Framework_name +'_'+ dataset_name + '_'+ net_name + '_BS_' + str(batch_size) +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSupervised_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_Affine_'+str(affine)+'/'

                load_path1 = dir_checkpoint + 'CP_epoch16.pth'
 
                net.module.load_state_dict(torch.load(load_path1 , map_location=device))
                train_net(net, epochs, batch_size, Learning_Rates_init[LR], device, video_path, output_video_path)

    except KeyboardInterrupt:
        
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
