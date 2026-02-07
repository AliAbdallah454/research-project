import os
import shutil

def delete_all_video_frames_folders(sessions_path: str):
    """
    Deletes every 'video_frames' folder under:
    sessions_path/<session>/<participant>/video_frames

    Skips .zip entries at the session level.
    """
    deleted = 0
    for session in os.listdir(sessions_path):

        if session.endswith(".zip") or '.DS_Store' in session:
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

            output_dir = os.path.join(participant_path, "video_frames")

            if os.path.isdir(output_dir):
                print("Deleting:", output_dir)
                shutil.rmtree(output_dir)
                deleted += 1

    print(f"Done. Deleted {deleted} 'video_frames' folders.")
    return deleted

sessions_path = "../data/Cornia/"
delete_all_video_frames_folders(sessions_path)
