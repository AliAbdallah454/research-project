import os
import cv2
import argparse

def get_frames_1hz(video_path, output_dir, resize=None, jpg_quality=95, show_progress=True):
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    saved = 0
    next_sec = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        processed += 1

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_sec = int(t_ms // 1000)

        if t_sec >= next_sec:
            if resize is not None:
                w, h = resize
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            out_path = os.path.join(output_dir, f"{saved:05d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1
            next_sec = t_sec + 1

        if show_progress and total_frames > 0:
            percent = 100 * processed / total_frames
            print(
                f"\rProcessed: {processed}/{total_frames} frames ({percent:.1f}%) | Saved: {saved}",
                end="",
                flush=True
            )

    cap.release()

    if show_progress:
        print()
        print(f"Done. Processed {processed} frames, saved {saved} frames.")

    return saved

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-path", required=True, help="Path to input video")
    p.add_argument("--out-path", required=True, help="Directory where frames will be stored")
    return p.parse_args()

args = parse_args()

video_path = args.video_path
frames_dir = args.out_path

get_frames_1hz(video_path, frames_dir)
