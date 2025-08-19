import cv2
import pandas as pd
import os

# --- CONFIGURATION ---
LABELS_FILE = 'labels.csv'
CALIBRATION_FILE = 'calibration.csv'
RAW_VIDEOS_FOLDER = 'data/raw_videos/'

# Define the frame offsets relative to the hop_start frame
BOARD_FRAME_OFFSET = -1 # UPDATED: Now looks 1 frame before takeoff
CONE_START_FRAME_OFFSET = -6
CONE_END_FRAME_OFFSET = -1

# --- Global variable to store click data ---
click_points = []

def mouse_callback(event, x, y, flags, param):
    """Handles mouse click events and provides visual feedback."""
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        
        # Use different colors for different clicks to guide the user
        if len(click_points) <= 2:
            color = (0, 255, 0) # Green for board points
            point_type = "Board"
        else:
            color = (0, 0, 255) # Red for cone points
            point_type = "Cone"
            
        cv2.circle(param['image'], (x, y), 7, color, -1)
        cv2.imshow(param['window_name'], param['image'])
        print(f"  - {point_type} Point {len(click_points)} registered at ({x}, {y})")

def get_clicks_for_frame(video_path, frame_num, instruction_text, num_clicks=1):
    """Opens a specific frame, displays instructions, and waits for clicks."""
    global click_points
    click_points = [] # Reset clicks for each new frame
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  - ERROR: Could not read frame {frame_num}. Skipping.")
        return None, None

    window_name = f"Calibration Frame: {frame_num}"
    cv2.namedWindow(window_name)
    callback_params = {'image': frame.copy(), 'window_name': window_name}
    cv2.setMouseCallback(window_name, mouse_callback, callback_params)

    print(instruction_text)
    cv2.imshow(window_name, callback_params['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(click_points) == num_clicks:
        return click_points, frame
    else:
        print(f"  - WARNING: Expected {num_clicks} click(s), but got {len(click_points)}. Please try again for this video.")
        return None, None

def run_advanced_calibration():
    """Main function to orchestrate the multi-step calibration process."""
    try:
        labels_df = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Labels file not found at '{LABELS_FILE}'.")
        return

    # Create calibration file with header if it doesn't exist
    if not os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'w') as f:
            f.write("video_name,board_p1_x,board_p1_y,board_p2_x,board_p2_y,board_frame,cone_start_x,cone_start_y,cone_start_frame,cone_end_x,cone_end_y,cone_end_frame\n")
    
    calibration_df = pd.read_csv(CALIBRATION_FILE)
    calibrated_videos = calibration_df['video_name'].tolist()

    print("--- Starting Advanced Interactive Video Calibration ---")
    
    for index, row in labels_df.iterrows():
        video_name = row['video_name']
        if video_name in calibrated_videos:
            print(f"\nSkipping '{video_name}' (already calibrated).")
            continue

        video_path = os.path.join(RAW_VIDEOS_FOLDER, video_name)
        if not os.path.exists(video_path):
            print(f"\nWARNING: Video for '{video_name}' not found. Skipping.")
            continue
        
        print(f"\n--- Calibrating: {video_name} ---")
        hop_start_frame = int(row['hop_start'])

        # --- Part 1: Calibrate Board Width ---
        board_frame = max(0, hop_start_frame + BOARD_FRAME_OFFSET)
        board_points, _ = get_clicks_for_frame(
            video_path, board_frame,
            "1. Click LEFT corner of board, then RIGHT corner. Press any key.", 2
        )
        if board_points is None: continue

        # --- Part 2: Calibrate Cone Start Position ---
        cone_start_frame = max(0, hop_start_frame + CONE_START_FRAME_OFFSET)
        cone_start_points, _ = get_clicks_for_frame(
            video_path, cone_start_frame,
            "2. Click the tip of the cone. Press any key.", 1
        )
        if cone_start_points is None: continue

        # --- Part 3: Calibrate Cone End Position ---
        cone_end_frame = max(1, hop_start_frame + CONE_END_FRAME_OFFSET)
        cone_end_points, _ = get_clicks_for_frame(
            video_path, cone_end_frame,
            "3. Click the tip of the SAME cone again. Press any key.", 1
        )
        if cone_end_points is None: continue

        # --- Part 4: Save all data ---
        bp1, bp2 = board_points[0], board_points[1]
        csp = cone_start_points[0]
        cep = cone_end_points[0]

        new_calib_data = {
            'video_name': [video_name],
            'board_p1_x': [bp1[0]], 'board_p1_y': [bp1[1]],
            'board_p2_x': [bp2[0]], 'board_p2_y': [bp2[1]],
            'board_frame': [board_frame],
            'cone_start_x': [csp[0]], 'cone_start_y': [csp[1]],
            'cone_start_frame': [cone_start_frame],
            'cone_end_x': [cep[0]], 'cone_end_y': [cep[1]],
            'cone_end_frame': [cone_end_frame]
        }
        
        new_df = pd.DataFrame(new_calib_data)
        new_df.to_csv(CALIBRATION_FILE, mode='a', header=False, index=False)
        print(f"  -> All calibration data for '{video_name}' saved successfully!")

if __name__ == "__main__":
    run_advanced_calibration()
    print("\n--- Calibration process finished. ---")
