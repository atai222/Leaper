import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# You need to know the FPS of your video. 
# Most online videos are 30 or 60. You can get the exact value
# from the 'fps' variable in your 'process_video.py' script.
FPS = 30 
TIME_PER_FRAME = 1 / FPS

# --- SCRIPT ---

# 1. Load the dataset with phase information
try:
    df = pd.read_csv('triple_jump_with_phases.csv')
except FileNotFoundError:
    print("Error: 'triple_jump_with_phases.csv' not found.")
    print("Please run the 'find_phases.py' script first.")
else:
    # 2. Calculate Center of Mass (CoM) for each frame
    df['CoM_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
    df['CoM_y'] = (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2

    # 3. Calculate the change in position (displacement) between frames
    # The .diff() method is perfect for this, it calculates the difference 
    # between an element and the previous element in the series.
    df['delta_CoM_x'] = df['CoM_x'].diff()
    df['delta_CoM_y'] = df['CoM_y'].diff()

    # 4. Calculate Velocity components
    # Velocity = Displacement / Time
    df['velocity_x'] = df['delta_CoM_x'] / TIME_PER_FRAME
    df['velocity_y'] = df['delta_CoM_y'] / TIME_PER_FRAME

    # 5. Calculate overall speed
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # The first row will have no velocity, so we fill it with 0
    df.fillna(0, inplace=True)

    # 6. Save the new, enriched dataset
    df.to_csv('triple_jump_with_velocity.csv', index=False)

    print("Successfully calculated velocity and speed!")
    print("New file saved: 'triple_jump_with_velocity.csv'")
    
    # Display the new columns for verification
    print("\nHere's a sample of the new data:")
    print(df[['frame', 'phase', 'velocity_x', 'velocity_y', 'speed']].head(10))

    # --- How to use this data next ---
    print("\nAverage speed per phase:")
    # Group by the 'phase' column and calculate the mean speed for each phase
    avg_speed_per_phase = df.groupby('phase')['speed'].mean()
    print(avg_speed_per_phase)