from pathlib import Path
import subprocess

def resize_video_ffmpeg(
    in_path: Path,
    out_path: Path,
    target_wh=(640, 360),
    mode: str = "fit",   # "fit" | "pad" | "crop" | "exact"
    crf: int = 23,       # lower = better quality, larger files; 18-28 typical
    preset: str = "medium",
    overwrite: bool = False,
):
    """
    Resize a video with ffmpeg.

    Modes:
      - "fit": keep aspect ratio, fit *within* target box; output dims may be <= target (flexible).
      - "pad": keep aspect ratio, fit within target, then pad to exactly target.
      - "crop": keep aspect ratio, scale so it covers target, then center-crop to exactly target.
      - "exact": force exactly target (may distort).

    Requires ffmpeg installed and available in PATH.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = target_wh

    if mode == "fit":
        # Scale down to fit within WxH, keep aspect ratio; allow smaller dim (flexible output size).
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease"
    elif mode == "pad":
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        )
    elif mode == "crop":
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    elif mode == "exact":
        vf = f"scale={w}:{h}"
    else:
        raise ValueError("mode must be one of: fit, pad, crop, exact")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(in_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "128k",
    ]

    if overwrite:
        cmd.insert(1, "-y")
    else:
        cmd.insert(1, "-n")  # don't overwrite

    cmd.append(str(out_path))

    subprocess.run(cmd, check=True)


def batch_resize_cornia_videos(
    root_dir: str | Path,
    target_wh=(640, 360),
    mode: str = "fit",
    out_name: str = "video_640x360.mp4",
    overwrite: bool = False,
):
    """
    Walk:
      root_dir/Cornia/Session*_Light/Participant*/video.mp4

    Writes resized video to the same participant folder with name `out_name`.
    """
    root_dir = Path(root_dir)
    cornia_dir = root_dir / "Cornia"
    if not cornia_dir.exists():
        raise FileNotFoundError(f"Not found: {cornia_dir}")

    video_paths = []
    for session_dir in sorted(cornia_dir.glob("Session*_Light")):
        if not session_dir.is_dir():
            continue
        for participant_dir in sorted(session_dir.glob("Participant*")):
            if not participant_dir.is_dir():
                continue
            in_vid = participant_dir / "video.mp4"
            if in_vid.exists():
                video_paths.append(in_vid)

    print(f"Found {len(video_paths)} videos.")
    for in_vid in video_paths:
        out_vid = in_vid.parent / out_name
        try:
            resize_video_ffmpeg(
                in_path=in_vid,
                out_path=out_vid,
                target_wh=target_wh,
                mode=mode,
                overwrite=overwrite,
            )
            print(f"OK  {in_vid} -> {out_vid}")
        except subprocess.CalledProcessError as e:
            print(f"FAIL {in_vid} ({e})")


# Example usage:
batch_resize_cornia_videos(
    root_dir="./data",
    target_wh=(640, 360),
    mode="fit",                 # flexible output size
    out_name="video_640x360.mp4",
    overwrite=False,
)
