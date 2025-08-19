import pandas as pd
import numpy as np
import os
import cv2
import joblib
from tqdm import tqdm
import mediapipe as mp

# --- CONFIGURATION ---
PROCESSED_CSVS_FOLDER = 'data/processed_csvs/'
RAW_VIDEOS_FOLDER = 'data/raw_videos/'
ANNOTATED_VIDEOS_FOLDER = 'reports/annotated_videos/' # Folder for the output clips
MODEL_FILE = 'models/distance_predictor_model.pkl'
BOARD_WIDTH_M = 0.2
VIDEO_FPS = 30

# --- HELPER FUNCTIONS ---

def process_video_if_needed(video_filename):
    """
    Checks if a video has been processed. If not, runs MediaPipe and saves
    both a landmark CSV and an annotated video clip for labeling.
    """
    base_filename = os.path.splitext(video_filename)[0]
    input_video_path = os.path.join(RAW_VIDEOS_FOLDER, video_filename)
    output_csv_path = os.path.join(PROCESSED_CSVS_FOLDER, f"{base_filename}.csv")
    output_video_path = os.path.join(ANNOTATED_VIDEOS_FOLDER, f"{base_filename}_annotated.mp4") # Path for annotated clip

    if os.path.exists(output_csv_path):
        print(f"'{video_filename}' has already been processed. Using existing data.")
        return True

    print(f"First time analyzing '{video_filename}'. Processing with MediaPipe...")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file at '{input_video_path}'")
        return False

    # --- Setup for creating the annotated video ---
    os.makedirs(ANNOTATED_VIDEOS_FOLDER, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    landmark_data = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_counter in tqdm(range(total_frames), desc="Extracting landmarks"):
        ret, frame = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks on the original frame for the video
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
            # Store landmark data for the CSV
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = {'frame': frame_counter}
            for i, lm in enumerate(landmarks):
                landmark_name = mp.solutions.pose.PoseLandmark(i).name
                frame_landmarks[f'{landmark_name}_x'] = lm.x
                frame_landmarks[f'{landmark_name}_y'] = lm.y
            landmark_data.append(frame_landmarks)
        
        # Add frame number to the video
        cv2.putText(frame, f"Frame: {frame_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        writer.write(frame) # Write the annotated frame to the new video file
    
    cap.release()
    writer.release() # Finalize the annotated video
    mp_pose.close()

    if landmark_data:
        df = pd.DataFrame(landmark_data)
        os.makedirs(PROCESSED_CSVS_FOLDER, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Landmark extraction complete. Data saved to '{output_csv_path}'")
        print(f"Annotated video for labeling saved to '{output_video_path}'")
        return True
    else:
        print("ERROR: No landmarks were detected in the video.")
        return False

def get_user_input(prompt, input_type=str):
    """A robust function to get typed input from the user."""
    while True:
        try:
            return input_type(input(prompt))
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")

def get_click_for_frame(video_path, frame_num, instruction_text):
    """Displays a frame and waits for a single user click."""
    click_point = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)
            cv2.circle(param['image'], (x, y), 7, (0, 255, 0), -1)
            cv2.imshow(param['window_name'], param['image'])
            print(f"  - Point registered at ({x}, {y})")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret: return None
    
    window_name = f"Calibration Frame: {frame_num}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, {'image': frame.copy(), 'window_name': window_name})
    print(instruction_text)
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return click_point

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    if any(val is None for val in [a, b, c]) or any(np.isnan(val) for val in np.concatenate([a,b,c])): return np.nan
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# --- MAIN ANALYSIS FUNCTION ---
def analyze_jump():
    # 1. Get Inputs from User
    video_filename = get_user_input("Enter the filename of the video clip to analyze: ")
    
    if not process_video_if_needed(video_filename):
        return

    total_distance_m = get_user_input("\nEnter the known total distance of the jump (in meters): ", float)
    
    print("\nWatch the annotated video in 'reports/annotated_videos/' to find the frames.")
    print("Enter the start and end frames for each phase:")
    phases_frames = {}
    for phase in ['hop', 'step', 'jump']:
        start = get_user_input(f"  - {phase.capitalize()} start frame: ", int)
        end = get_user_input(f"  - {phase.capitalize()} end frame: ", int)
        phases_frames[phase] = {'start': start, 'end': end}

    # 2. Load necessary files
    try:
        model = joblib.load(MODEL_FILE)
        df = pd.read_csv(os.path.join(PROCESSED_CSVS_FOLDER, f"{os.path.splitext(video_filename)[0]}.csv"))
        video_path = os.path.join(RAW_VIDEOS_FOLDER, video_filename)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nERROR: Could not find a required file. Details: {e}")
        return

    # 3. Perform Live Calibration
    print("\n--- Starting Live Calibration ---")
    board_frame = max(0, phases_frames['hop']['start'] - 1)
    print("First, we'll calibrate the board width.")
    bp1 = get_click_for_frame(video_path, board_frame, "Click the LEFT corner of the board.")
    bp2 = get_click_for_frame(video_path, board_frame, "Click the RIGHT corner of the board.")
    if not bp1 or not bp2: print("Board calibration failed."); return

    print("\nNow, we'll calibrate the camera pan by tracking a static object.")
    cone_start_frame = max(0, phases_frames['hop']['start'] - 6)
    cone_end_frame = max(1, phases_frames['hop']['start'] - 1)
    csp = get_click_for_frame(video_path, cone_start_frame, "Click a static object (e.g., cone tip).")
    cep = get_click_for_frame(video_path, cone_end_frame, "Click the SAME static object again.")
    if not csp or not cep: print("Camera pan calibration failed."); return

    # 4. Calculate Features based on calibration and inputs
    board_pixel_dist = np.linalg.norm(np.array(bp1) - np.array(bp2))
    pixels_per_meter = board_pixel_dist / BOARD_WIDTH_M
    
    time_elapsed_s = (cone_end_frame - cone_start_frame) / VIDEO_FPS
    camera_pan_pixels = cep[0] - csp[0]
    camera_pan_speed_pps = camera_pan_pixels / time_elapsed_s

    df['CoM_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
    athlete_com_start = df.loc[cone_start_frame]['CoM_x']
    athlete_com_end = df.loc[cone_end_frame]['CoM_x']
    athlete_pixel_change = athlete_com_end - athlete_com_start
    athlete_speed_pps = athlete_pixel_change / time_elapsed_s

    true_speed_pps = athlete_speed_pps - camera_pan_speed_pps
    entry_velocity_mps = true_speed_pps / pixels_per_meter

    # 5. Predict for each phase
    predictions = {}
    for phase in ['hop', 'step', 'jump']:
        start_frame = phases_frames[phase]['start']
        end_frame = phases_frames[phase]['end']
        phase_df = df.iloc[start_frame:end_frame + 1]

        features = {}
        features['entry_velocity_mps'] = entry_velocity_mps
        features['contact_time_s'] = (end_frame - start_frame) / VIDEO_FPS
        knee_angles = [calculate_angle(
            [fr['RIGHT_HIP_x'], fr['RIGHT_HIP_y']], [fr['RIGHT_KNEE_x'], fr['RIGHT_KNEE_y']], [fr['RIGHT_ANKLE_x'], fr['RIGHT_ANKLE_y']]
        ) for _, fr in phase_df.iterrows()]
        features['min_knee_angle'] = np.nanmin(knee_angles) if knee_angles else np.nan
        
        feature_df_for_pred = pd.DataFrame([features])[model.feature_names_in_]
        predictions[phase] = model.predict(feature_df_for_pred)[0]

    # 6. Display Report
    print("\n--- JUMP ANALYSIS REPORT ---")
    print(f"Analysis for: {video_filename}")
    print(f"Known Total Distance: {total_distance_m:.2f} m")
    print(f"Calculated Entry Velocity: {entry_velocity_mps:.2f} m/s")
    
    print("\n--- Raw Model Predictions ---")
    raw_total = sum(predictions.values())
    for phase, dist in predictions.items():
        print(f"  - Predicted {phase.capitalize()} Distance: {dist:.2f} m")
    print(f"  - Predicted Total Distance: {raw_total:.2f} m")

    print("\n--- Normalized Predictions ---")
    if raw_total > 0:
        for phase, dist in predictions.items():
            normalized_dist = (dist / raw_total) * total_distance_m
            print(f"  - Normalized {phase.capitalize()} Distance: {normalized_dist:.2f} m")
    print("---------------------------------")

if __name__ == '__main__':
    analyze_jump()
