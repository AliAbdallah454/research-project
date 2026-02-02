import os
import pandas as pd

def normalize_results_manual_in_tree(
    sessions_path: str = "./data/Cornia/",
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

    cols_x = ["Center-X Zone 1", "Center-X Zone 2"]
    cols_y = ["Center-Y Zone 1", "Center-Y Zone 2"]
    cols_r = ["Radius Zone 1", "Radius Zone 2"]
    target_cols = cols_x + cols_y + cols_r

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

            # Read TSV
            df = pd.read_csv(in_path, sep="\t")

            # Basic validation: only normalize columns that exist
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                print(f"Skipping {in_path} (missing columns: {missing})")
                continue

            # Convert to numeric (coerce invalid to NaN)
            for c in target_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Apply normalization
            for c in cols_x:
                df[c] = df[c] / float(img_w)
            for c in cols_y:
                df[c] = df[c] / float(img_h)
            r_norm = float(min(img_w, img_h))
            for c in cols_r:
                df[c] = df[c] / r_norm

            out_path = os.path.join(participant_path, output_filename)
            df.to_csv(out_path, sep="\t", index=False)

            processed += 1

    return processed


# Example usage:
n = normalize_results_manual_in_tree("./data/Cornia/", image_size=(1920, 1080))
print("Processed:", n)
