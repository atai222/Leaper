import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
LABELS_FILE = 'labels.csv'
PROCESSED_CSVS_FOLDER = 'data/processed_csvs/'
OUTPUT_TRAINING_FILE = 'training_data.csv'
VIDEO_FPS = 30 # IMPORTANT: Assume all videos are 30 FPS. Change if necessary.

# --- HELPER FUNCTION ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # Handle potential missing data
    if any(val is None for val in [a, b, c]) or any(np.isnan(val) for val in np.concatenate([a,b,c])):
        return np.nan
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clip to avoid math errors with values slightly out of [-1, 1] range
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- MAIN SCRIPT ---

def create_feature_dataset():
    """
    Reads labeled data and processed landmark CSVs to generate a feature set
    for training a regression model.
    """
    try:
        labels_df = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Labels file not found at '{LABELS_FILE}'")
        return

    all_phase_features = []

    print("Starting feature engineering process...")
    # Loop through each jump in the labels file
    for index, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing Jumps"):
        video_name = row['video_name']
        csv_path = os.path.join(PROCESSED_CSVS_FOLDER, os.path.splitext(video_name)[0] + '.csv')

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            tqdm.write(f"WARNING: Data file for '{video_name}' not found. Skipping.")
            continue

        # --- Pre-calculate velocity for the whole jump ---
        df['CoM_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
        df['velocity_x'] = df['CoM_x'].diff() * VIDEO_FPS # pixels/sec
        df.fillna(0, inplace=True)

        # Define the phases to process
        phases = ['hop', 'step', 'jump']
        for phase in phases:
            start_frame = int(row[f'{phase}_start'])
            end_frame = int(row[f'{phase}_end'])
            
            # Isolate the data for the current phase (ground contact)
            phase_df = df.iloc[start_frame:end_frame + 1].copy()
            if phase_df.empty:
                continue

            # --- Feature Calculation ---
            features = {}
            features['video_name'] = video_name
            features['phase_type'] = phase
            
            # 1. Contact Time
            features['contact_time_s'] = (end_frame - start_frame) / VIDEO_FPS

            # 2. Horizontal Velocity Loss (Force Loss)
            v_in = df.loc[max(0, start_frame - 1)]['velocity_x']
            v_out = df.loc[min(len(df) - 1, end_frame + 1)]['velocity_x']
            features['horiz_velo_loss'] = v_in - v_out
            features['entry_velocity'] = v_in

            # 3. Minimum Knee Angle
            knee_angles = []
            for i, frame_row in phase_df.iterrows():
                # Assuming right-leg dominant for now
                hip = [frame_row['RIGHT_HIP_x'], frame_row['RIGHT_HIP_y']]
                knee = [frame_row['RIGHT_KNEE_x'], frame_row['RIGHT_KNEE_y']]
                ankle = [frame_row['RIGHT_ANKLE_x'], frame_row['RIGHT_ANKLE_y']]
                knee_angles.append(calculate_angle(hip, knee, ankle))
            
            features['min_knee_angle'] = np.nanmin(knee_angles) if knee_angles else np.nan

            # 4. Center of Mass Lowering
            # Find min CoM height during contact vs. height just before contact
            com_y_in = df.loc[max(0, start_frame - 1)]['LEFT_HIP_y'] # Use hip as proxy
            min_com_y_contact = phase_df['LEFT_HIP_y'].max() # Max y is lowest point
            features['com_lowering'] = min_com_y_contact - com_y_in # In pixels

            # --- Target Variable ---
            features['distance_m'] = row[f'{phase}_dist_m']

            all_phase_features.append(features)

    # Create and save the final training DataFrame
    if all_phase_features:
        training_df = pd.DataFrame(all_phase_features)
        training_df.to_csv(OUTPUT_TRAINING_FILE, index=False)
        print("\n--- Feature engineering complete! ---")
        print(f"Successfully created '{OUTPUT_TRAINING_FILE}' with {len(training_df)} samples.")
        print("\nHere's a sample of your new training data:")
        print(training_df.head())
    else:
        print("No features were generated. Check your labels and data files.")


if __name__ == "__main__":
    create_feature_dataset()

