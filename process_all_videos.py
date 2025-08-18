import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm # A library to create smart progress bars

# --- HELPER FUNCTION (from previous scripts) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- CONFIGURATION ---
RAW_VIDEOS_FOLDER = 'data/raw_videos/'
PROCESSED_CSVS_FOLDER = 'data/processed_csvs/'
ANNOTATED_VIDEOS_FOLDER = 'reports/annotated_videos/'

# --- SCRIPT ---

def batch_process_videos():
    """
    Loops through all videos in a folder, runs MediaPipe pose estimation,
    and saves the landmark data to CSV files and creates annotated videos.
    """
    # 1. Ensure all necessary output directories exist
    for folder in [PROCESSED_CSVS_FOLDER, ANNOTATED_VIDEOS_FOLDER]:
        if not os.path.exists(folder):
            print(f"Creating directory: {folder}")
            os.makedirs(folder)

    # 2. Get the list of video files to process
    try:
        video_files = [f for f in os.listdir(RAW_VIDEOS_FOLDER) if f.endswith('.mp4')]
        if not video_files:
            print(f"No .mp4 files found in '{RAW_VIDEOS_FOLDER}'. Please add video clips.")
            return
    except FileNotFoundError:
        print(f"ERROR: Raw videos folder not found at '{RAW_VIDEOS_FOLDER}'")
        return
        
    print(f"Found {len(video_files)} videos to process.")

    # 3. Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 4. Loop through each video file with a progress bar
    for video_filename in tqdm(video_files, desc="Processing Videos"):
        base_filename = os.path.splitext(video_filename)[0]
        input_video_path = os.path.join(RAW_VIDEOS_FOLDER, video_filename)
        output_csv_path = os.path.join(PROCESSED_CSVS_FOLDER, f"{base_filename}.csv")
        output_video_path = os.path.join(ANNOTATED_VIDEOS_FOLDER, f"{base_filename}_annotated.mp4")

        # --- Efficiency Check: Skip if already processed ---
        if os.path.exists(output_csv_path):
            #tqdm.write(f"Skipping '{video_filename}', CSV already exists.")
            continue

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            tqdm.write(f"Error opening video file: {video_filename}")
            continue

        # Setup VideoWriter for the annotated output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        landmark_data = []
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # --- Store all landmark data for the CSV ---
                frame_landmarks = {'frame': frame_counter}
                for i, lm in enumerate(landmarks):
                    landmark_name = mp_pose.PoseLandmark(i).name
                    frame_landmarks[f'{landmark_name}_x'] = lm.x
                    frame_landmarks[f'{landmark_name}_y'] = lm.y
                    frame_landmarks[f'{landmark_name}_z'] = lm.z
                    frame_landmarks[f'{landmark_name}_visibility'] = lm.visibility
                landmark_data.append(frame_landmarks)

                # --- Draw on the frame for the annotated video ---
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Add frame number to the video
            cv2.putText(image, f"Frame: {frame_counter}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            writer.write(image)
            frame_counter += 1

        # Release resources for this video
        cap.release()
        writer.release()

        # Save the collected data to a CSV file
        if landmark_data:
            df = pd.DataFrame(landmark_data)
            df.to_csv(output_csv_path, index=False)
    
    # Clean up MediaPipe
    pose.close()


if __name__ == "__main__":
    batch_process_videos()
    print("\n--- Batch processing complete. ---")
    print(f"Processed CSVs are in: '{PROCESSED_CSVS_FOLDER}'")
    print(f"Annotated videos for labeling are in: '{ANNOTATED_VIDEOS_FOLDER}'")

