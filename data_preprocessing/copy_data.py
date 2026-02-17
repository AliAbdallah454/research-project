import os
import shutil

import argparse

def copy_processed_data(
    sessions_path: str = "../data/Cornia/",
    out_root: str = "../data/processed_data",
    frames_dir_name: str = "video_frames",
    results_file_name: str = "normalized_results_manual.txt",
) -> int:
    """
    Creates a mirror directory tree under out_root:

    out_root/
      SessionX/
        ParticipantY/
          video_frames/   (copied folder)
          normalized_manual_results.txt (copied file)

    Skips sessions like .zip and .DS_Store. Skips non-Participant folders.
    Returns the number of participants successfully copied.
    """
    os.makedirs(out_root, exist_ok=True)
    copied = 0

    for session in os.listdir(sessions_path):
        if session.endswith(".zip") or session == ".DS_Store":
            continue

        session_path = os.path.join(sessions_path, session)
        if not os.path.isdir(session_path):
            continue

        print("Working on:", session)

        for participant in os.listdir(session_path):
            if "Participant" not in participant:
                continue

            participant_path = os.path.join(session_path, participant)
            if not os.path.isdir(participant_path):
                continue

            src_frames = os.path.join(participant_path, frames_dir_name)
            src_results = os.path.join(participant_path, results_file_name)

            # Must exist to copy
            if not os.path.isdir(src_frames):
                print(f"\tSkipping {participant} (missing folder: {frames_dir_name})")
                continue
            if not os.path.isfile(src_results):
                print(f"\tSkipping {participant} (missing file: {results_file_name})")
                continue

            # Destination paths
            dst_participant_dir = os.path.join(out_root, session, participant)
            dst_frames = os.path.join(dst_participant_dir, frames_dir_name)
            dst_results = os.path.join(dst_participant_dir, results_file_name)

            os.makedirs(dst_participant_dir, exist_ok=True)

            # Copy folder (merge if exists)
            if os.path.exists(dst_frames):
                print(f"\t{dst_frames} already exists -> merging/overwriting files")
            shutil.copytree(src_frames, dst_frames, dirs_exist_ok=True)

            # Copy file (overwrite)
            shutil.copy2(src_results, dst_results)

            print(f"\tCopied -> {dst_participant_dir}")
            copied += 1

    return copied

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to dataset")
parser.add_argument("--output-path", required=True, help="Path to output directory")
args = parser.parse_args()

sessions_path = args.data_path
output_path = args.output_path

# Example usage:
n = copy_processed_data(
    sessions_path=sessions_path,
    out_root=output_path,
    frames_dir_name="video_frames",
    results_file_name="normalized_results_manual.txt"
)
print("Participants copied:", n)
