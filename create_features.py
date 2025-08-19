import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
LABELS_FILE = 'labels.csv'
CALIBRATION_FILE = 'calibration.csv'
PROCESSED_CSVS_FOLDER = 'data/processed_csvs/'
OUTPUT_TRAINING_FILE = 'training_data.csv'
VIDEO_FPS = 30
BOARD_WIDTH_M = 0.2  # The known width of the takeoff board's short side

# --- HELPER FUNCTION ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    if any(val is None for val in [a, b, c]) or any(np.isnan(val) for val in np.concatenate([a,b,c])): return np.nan
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# --- MAIN SCRIPT ---

def create_final_feature_dataset():
    """
    Generates the final feature set for regression using the advanced,
    4-point calibration data to calculate true velocity.
    """
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        calibration_df = pd.read_csv(CALIBRATION_FILE).set_index('video_name')
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return

    all_phase_features = []

    print("Starting final feature engineering process...")
    for index, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing Jumps"):
        video_name = row['video_name']
        
        try:
            df = pd.read_csv(os.path.join(PROCESSED_CSVS_FOLDER, f"{os.path.splitext(video_name)[0]}.csv"))
            calib_data = calibration_df.loc[video_name]
        except (FileNotFoundError, KeyError):
            tqdm.write(f"WARNING: Data or calibration for '{video_name}' not found. Skipping.")
            continue

        # --- 1. Calculate Pixels-Per-Meter Ratio from board clicks ---
        board_p1 = np.array([calib_data['board_p1_x'], calib_data['board_p1_y']])
        board_p2 = np.array([calib_data['board_p2_x'], calib_data['board_p2_y']])
        board_pixel_dist = np.linalg.norm(board_p1 - board_p2)
        pixels_per_meter = board_pixel_dist / BOARD_WIDTH_M

        # --- 2. Calculate True Entry Velocity using Cone Tracking ---
        cone_start_frame = int(calib_data['cone_start_frame'])
        cone_end_frame = int(calib_data['cone_end_frame'])
        
        # Ensure the frames are within the bounds of the dataframe
        if cone_start_frame >= len(df) or cone_end_frame >= len(df):
            tqdm.write(f"WARNING: Cone frames out of bounds for '{video_name}'. Skipping velocity calc.")
            continue
            
        time_elapsed_s = (cone_end_frame - cone_start_frame) / VIDEO_FPS
        if time_elapsed_s <= 0:
            tqdm.write(f"WARNING: Invalid time window for '{video_name}'. Skipping velocity calc.")
            continue

        # Calculate camera pan speed
        camera_pan_pixels = calib_data['cone_end_x'] - calib_data['cone_start_x']
        camera_pan_speed_pps = camera_pan_pixels / time_elapsed_s # pixels / sec

        # Calculate athlete's speed relative to the camera frame
        df['CoM_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
        athlete_com_start = df.loc[cone_start_frame]['CoM_x']
        athlete_com_end = df.loc[cone_end_frame]['CoM_x']
        athlete_pixel_change = athlete_com_end - athlete_com_start
        athlete_speed_pps = athlete_pixel_change / time_elapsed_s

        # Calculate true ground speed
        true_speed_pps = athlete_speed_pps - camera_pan_speed_pps
        entry_velocity_mps = true_speed_pps / pixels_per_meter

        # --- 3. Process each phase for this jump ---
        for phase in ['hop', 'step', 'jump']:
            start_frame = int(row[f'{phase}_start'])
            end_frame = int(row[f'{phase}_end'])
            phase_df = df.iloc[start_frame:end_frame + 1].copy()
            if phase_df.empty: continue

            features = {}
            features['video_name'] = video_name
            features['phase_type'] = phase
            features['entry_velocity_mps'] = entry_velocity_mps

            # Feature: Contact Time
            features['contact_time_s'] = (end_frame - start_frame) / VIDEO_FPS

            # Feature: Minimum Knee Angle (assuming right-leg dominant)
            knee_angles = [calculate_angle(
                [fr['RIGHT_HIP_x'], fr['RIGHT_HIP_y']],
                [fr['RIGHT_KNEE_x'], fr['RIGHT_KNEE_y']],
                [fr['RIGHT_ANKLE_x'], fr['RIGHT_ANKLE_y']]
            ) for _, fr in phase_df.iterrows()]
            features['min_knee_angle'] = np.nanmin(knee_angles) if knee_angles else np.nan

            # Target Variable
            features['distance_m'] = row[f'{phase}_dist_m']
            all_phase_features.append(features)

    # --- 4. Save Final Dataset ---
    if all_phase_features:
        training_df = pd.DataFrame(all_phase_features)
        training_df.to_csv(OUTPUT_TRAINING_FILE, index=False)
        print(f"\n--- Feature engineering complete! ---")
        print(f"Successfully created '{OUTPUT_TRAINING_FILE}' with {len(training_df)} samples.")
        print("\nSample of your new training data:")
        print(training_df.head())
    else:
        print("No features were generated.")

if __name__ == "__main__":
    create_final_feature_dataset()
