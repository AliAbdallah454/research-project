import os
import cv2

import argparse

def get_frames_1hz(video_path, output_dir, resize=None, jpg_quality=95):

    if resize is not None:
        if (not isinstance(resize, (tuple, list))) or len(resize) != 2:
            raise ValueError("resize must be None or (width, height)")
        w, h = resize
        if w <= 0 or h <= 0:
            raise ValueError("resize dimensions must be > 0")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    saved = 0
    next_sec = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t_ms is None:
            continue
        t_sec = int(t_ms // 1000)

        if t_sec >= next_sec:
            if resize is not None:
                w, h = resize
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            out_path = os.path.join(output_dir, f"img{saved}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1
            next_sec = t_sec + 1

    cap.release()
    return saved

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to dataset")
args = parser.parse_args()

sessions_path = args.data_path

for session in os.listdir(sessions_path):

  if '.zip' in session or '.DS_Store' in session:
    continue

  print("Working on: ", session)
  session_path = os.path.join(sessions_path, session)
  for participant in os.listdir(session_path):

    if "Participant" not in participant:
       continue

    participant_path = os.path.join(session_path, participant)

    print("\tWorking on: ", participant_path)

    video_path = os.path.join(participant_path, "video.mp4")
    text_file_path = os.path.join(participant_path, "results_manual.txt")
    output_dir = os.path.join(participant_path, "video_frames")

    if os.path.exists(output_dir):
       print(output_dir, " Already Exists ...")
       continue

    if not os.path.exists(video_path):
      continue
    if not os.path.exists(text_file_path):
      continue

    frames_saved = get_frames_1hz(video_path, output_dir, (640, 360))
    # if frames_saved != get_number_of_frames(text_file_path):
    #   print(f"\More Frames in {participant} ...")