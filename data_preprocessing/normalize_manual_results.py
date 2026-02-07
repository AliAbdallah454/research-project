import os
import pandas as pd

expected_cols = [
    "Time in s",
    "Defined zone",
    "Point number",
    "Center-X Zone 1",
    "Center-Y Zone 1",
    "Radius Zone 1",
    "Center-X Zone 2",
    "Center-Y Zone 2",
    "Radius Zone 2"
]

def normalize_results_manual_in_tree(
    sessions_path: str = "../data/Cornia/",
    image_size: tuple[int, int] = (1920, 1080),
    input_filename: str = "results_manual.txt",
    output_filename: str = "normalized_results_manual.txt",
) -> int:
    """
    Walks sessions_path and for each Participant*/ folder containing results_manual.txt,
    normalizes specific columns and saves to normalized_results_manual.txt (tab-separated).

    Normalization:
      - Center-X / image_width
      - Center-Y / image_height
      - Radius   / min(image_width, image_height)

    Returns:
      number of files processed.
    """
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image_size must be positive (width, height)")

    processed = 0

    for session in os.listdir(sessions_path):
        if session.endswith(".zip") or session == ".DS_Store":
            continue

        session_path = os.path.join(sessions_path, session)
        if not os.path.isdir(session_path):
            continue

        for participant in os.listdir(session_path):
            
            if "Participant" not in participant:
                continue

            participant_path = os.path.join(session_path, participant)
            if not os.path.isdir(participant_path):
                continue

            in_path = os.path.join(participant_path, input_filename)
            if not os.path.exists(in_path):
                continue

            df = pd.read_csv(in_path, sep="\t", usecols=expected_cols)

            df["Center-X Zone 1"] = df["Center-X Zone 1"] / float(img_w)
            df["Center-Y Zone 1"] = df["Center-Y Zone 1"] / float(img_h)
            df["Radius Zone 1"] = df["Radius Zone 1"] / float(min(img_h, img_w))

            df["Center-X Zone 2"] = df["Center-X Zone 2"] / float(img_w)
            df["Center-Y Zone 2"] = df["Center-Y Zone 2"] / float(img_h)
            df["Radius Zone 2"] = df["Radius Zone 2"] / float(min(img_h, img_w))

            out_path = os.path.join(participant_path, output_filename)

            if os.path.exists(out_path):
                os.remove(out_path)

            df.to_csv(out_path, sep="\t", index=False)

            processed += 1

    return processed


# Example usage:
n = normalize_results_manual_in_tree("../data/Cornia/", image_size=(1920, 1080))
print("Processed:", n)
