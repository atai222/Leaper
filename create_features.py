import pandas as pd
import numpy as np

# --- CONFIGURATION ---
WINDOW_SIZE = 20  # How many frames to look at once.
DATA_FILE = 'triple_jump_analysis.csv'
LABEL_FILE = 'labels.csv' # File with start and end frames

# --- SCRIPT ---

# 1. Load data and labels
try:
    df = pd.read_csv(DATA_FILE)
    labels_df = pd.read_csv(LABEL_FILE)
except FileNotFoundError as e:
    print(f"Error: Make sure '{e.filename}' is in the same folder.")
else:
    # --- NEW: Get start and end frames from your labels file ---
    takeoff_start = labels_df['takeoff_start_frame'].iloc[0]
    takeoff_end = labels_df['takeoff_end_frame'].iloc[0]

    features_list = []
    labels_list = []

    # 2. Slide the window across the data
    for i in range(WINDOW_SIZE // 2, len(df) - WINDOW_SIZE // 2):
        start_index = i - WINDOW_SIZE // 2
        end_index = i + WINDOW_SIZE // 2
        
        window = df.iloc[start_index:end_index]
        left_ankle_y = window['LEFT_ANKLE_y']
        right_ankle_y = window['RIGHT_ANKLE_y']

        # 3. Calculate features for the window
        features = {
            'left_mean': left_ankle_y.mean(),
            'left_std': left_ankle_y.std(),
            'left_min': left_ankle_y.min(),
            'left_max': left_ankle_y.max(),
            'right_mean': right_ankle_y.mean(),
            'right_std': right_ankle_y.std(),
            'right_min': right_ankle_y.min(),
            'right_max': right_ankle_y.max(),
            'left_slope': left_ankle_y.iloc[-1] - left_ankle_y.iloc[0],
            'right_slope': right_ankle_y.iloc[-1] - right_ankle_y.iloc[0]
        }
        features_list.append(features)

        # 4. --- MODIFIED: Assign the label based on the range ---
        # If the center of our window is within the ground contact phase, label it 1.
        if takeoff_start <= i <= takeoff_end:
            labels_list.append(1)
        else:
            labels_list.append(0)

    # 5. Create the final training DataFrame
    training_df = pd.DataFrame(features_list)
    training_df['label'] = labels_list

    training_df.to_csv('training_data.csv', index=False)
    
    print("Successfully created feature set!")
    print(f"Saved to 'training_data.csv'. Total windows created: {len(training_df)}")
    print(f"Number of 'takeoff' windows (label 1): {sum(labels_list)}")
    print("\nSample of the training data:")
    print(training_df.head())
