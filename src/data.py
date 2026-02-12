import os
import pandas as pd
import random

from torch.utils.data import DataLoader

from typing import List

from src.helpers import read_manual_results
from src.processed_dataset import ProcessedDataset, train_tf, val_tf

def load_participants(root: str):

    participants = []

    for session in os.listdir(root):

        session_path = os.path.join(root, session)
        for participant in os.listdir(session_path):

            if '.zip' in participant or "Participant" not in participant:
                continue

            participant_path = os.path.join(session_path, participant)
            participants.append(participant_path)

    return participants

def get_input_target_lists(participants: List[str]):

    images = []
    targets = []

    for participant_path in participants:

        manual_data_df = read_manual_results(os.path.join(participant_path, "normalized_results_manual.txt"))
        images_path = os.path.join(participant_path, "video_frames")

        for image in os.listdir(images_path):

            image_time_stamp = int(image.split('.')[0][3:])
            if image_time_stamp < len(manual_data_df):
                info = manual_data_df.iloc[image_time_stamp]
                target = (
                    float(info['Center-X Zone 1']),
                    float(info['Center-Y Zone 1']),
                    float(info['Radius Zone 1']),

                    float(info['Center-X Zone 2']),
                    float(info['Center-Y Zone 2']),
                    float(info['Radius Zone 2'])
                )

                images.append(os.path.join(participant_path, "video_frames", image))
                targets.append(target)
                
    return images, targets

def get_loaders(root_path: str, batch_size:int = 16, seed: int = 43):

    participants = load_participants(root_path)
    participants = sorted(participants)

    rng = random.Random(seed)
    rng.shuffle(participants)

    n = len(participants)
    trainlim = int(0.8 * n)
    vallim = trainlim + int(0.1 * n)

    train_participants = participants[:trainlim]
    val_participants   = participants[trainlim:vallim]
    test_participants  = participants[vallim:]  # remainder

    assert len(participants) == len(train_participants) + len(val_participants) + len(test_participants), "INVALID"

    train_images, train_targets = get_input_target_lists(train_participants)
    val_images, val_targets = get_input_target_lists(val_participants)
    test_images, test_targets = get_input_target_lists(test_participants)


    train_ds = ProcessedDataset(train_images, train_targets, transform=train_tf)
    val_ds = ProcessedDataset(val_images, val_targets, transform=val_tf)
    test_ds = ProcessedDataset(test_images, test_targets, transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl  = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl