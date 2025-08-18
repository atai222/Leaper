import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import joblib # Used to load your saved model

# --- Helper Function to Calculate Angles (from process_video.py) ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- CONFIGURATION ---
# IMPORTANT: Update these filenames for your new video
NEW_VIDEO_FILENAME = 'Triple Jump World Record Slow Motion Jonathan Edwards 18.29 [rjT_JwRi0oA].mp4' 
MODEL_FILENAME = 'takeoff_detector_model.pkl'
OUTPUT_VIDEO_FILENAME = 'predicted_output.mp4'
WINDOW_SIZE = 20 # Must be the same as in your create_features.py script

# --- SCRIPT ---

# 1. Load the trained model
print(f"Loading model: {MODEL_FILENAME}")
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILENAME}' not found. Make sure you've run train_model.py")
    exit()

# 2. Process the new video to get landmark data
print(f"Processing new video: {NEW_VIDEO_FILENAME}")
video_path = os.path.join(os.getcwd(), NEW_VIDEO_FILENAME)
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

landmark_data = []
frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame_landmarks = {'frame': frame_counter}
        for i, lm in enumerate(landmarks):
            frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_x'] = lm.x
            frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_y'] = lm.y
        landmark_data.append(frame_landmarks)
    frame_counter += 1
cap.release()
pose.close()

if not landmark_data:
    print("Could not detect any poses in the new video.")
    exit()

df_new_video = pd.DataFrame(landmark_data)
print(f"Extracted data for {len(df_new_video)} frames from the new video.")

# 3. Create features for the new video data using the sliding window
print("Creating features for the new video...")
features_list = []
for i in range(WINDOW_SIZE // 2, len(df_new_video) - WINDOW_SIZE // 2):
    start_index = i - WINDOW_SIZE // 2
    end_index = i + WINDOW_SIZE // 2
    window = df_new_video.iloc[start_index:end_index]
    
    # Check if required columns exist before creating features
    if 'LEFT_ANKLE_y' in window and 'RIGHT_ANKLE_y' in window:
        left_ankle_y = window['LEFT_ANKLE_y']
        right_ankle_y = window['RIGHT_ANKLE_y']
        
        features = {
            'left_mean': left_ankle_y.mean(), 'left_std': left_ankle_y.std(),
            'left_min': left_ankle_y.min(), 'left_max': left_ankle_y.max(),
            'right_mean': right_ankle_y.mean(), 'right_std': right_ankle_y.std(),
            'right_min': right_ankle_y.min(), 'right_max': right_ankle_y.max(),
            'left_slope': left_ankle_y.iloc[-1] - left_ankle_y.iloc[0],
            'right_slope': right_ankle_y.iloc[-1] - right_ankle_y.iloc[0]
        }
        features_list.append(features)

features_df = pd.DataFrame(features_list)

# 4. Make predictions on the new features
print("Making predictions...")
# We use predict_proba to get the confidence score for each class
# We are interested in the probability of class '1' (takeoff)
predictions_proba = model.predict_proba(features_df)[:, 1]

# 5. Find the frame with the highest takeoff probability
highest_prob_index = np.argmax(predictions_proba)
# Adjust index to match the original frame number
predicted_takeoff_frame = highest_prob_index + (WINDOW_SIZE // 2) 
highest_prob_value = predictions_proba[highest_prob_index]

print(f"\n--- PREDICTION COMPLETE ---")
print(f"Predicted takeoff frame: {predicted_takeoff_frame}")
print(f"Model confidence: {highest_prob_value:.2%}")

# 6. Create the final annotated video
print(f"Creating annotated output video: {OUTPUT_VIDEO_FILENAME}")
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, fps, (frame_width, frame_height))

frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display text on the video. If the current frame is the predicted one, show it.
    prob_index = frame_counter - (WINDOW_SIZE // 2)
    if 0 <= prob_index < len(predictions_proba):
        current_prob = predictions_proba[prob_index]
        # If the probability is high (e.g., > 50%), label it as a takeoff phase
        if current_prob > 0.5:
            text = "PREDICTED TAKEOFF"
            color = (0, 255, 0) # Green
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

    # Display the frame number
    cv2.putText(frame, f"Frame: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    writer.write(frame)
    frame_counter += 1

cap.release()
writer.release()
print("--- DONE ---")
