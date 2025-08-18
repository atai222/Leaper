import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., a joint)."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex of the angle)
    c = np.array(c)  # End point

    # Calculate vectors from the mid point to the other two points
    ba = a - b
    bc = c - b

    # Calculate the dot product and the magnitude of the vectors
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Get the angle in radians and then convert to degrees
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# --- Main Video Processing ---

# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. Specify Video Paths
# IMPORTANT: Replace with the actual name of your downloaded video file.
input_filename = 'TRIPLE JUMP DOUBLE ARM TECHNIQUE [VPnfnvANgVQ].f609.mp4'
output_filename_video = 'triple_jump_output.mp4'
output_filename_csv = 'triple_jump_analysis.csv'

video_path = os.path.join(os.getcwd(), input_filename)

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print("Please make sure the video file is in the same directory as the script and the filename is correct.")
else:
    cap = cv2.VideoCapture(video_path)

    # --- Get video properties for VideoWriter ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- Initialize VideoWriter object ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename_video, fourcc, fps, (frame_width, frame_height))


    # 3. Data Storage
    landmark_data = []
    frame_counter = 0

    # 4. Loop Through Video Frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 5. Extract Landmarks and Calculate Features
        try:
            landmarks = results.pose_landmarks.landmark
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            frame_landmarks = {'frame': frame_counter}
            frame_landmarks['left_knee_angle'] = left_knee_angle
            frame_landmarks['right_knee_angle'] = right_knee_angle

            for i, lm in enumerate(landmarks):
                frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_x'] = lm.x
                frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_y'] = lm.y
                frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_z'] = lm.z
                frame_landmarks[f'{mp_pose.PoseLandmark(i).name}_visibility'] = lm.visibility
            
            landmark_data.append(frame_landmarks)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.putText(image, f"Left Knee Angle: {int(left_knee_angle)}", 
                        tuple(np.multiply(left_knee, [frame_width, frame_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            print(f"No landmarks detected in frame {frame_counter}")
            pass

        # --- NEW: Display the frame number on the video ---
        cv2.putText(image, f"Frame: {frame_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(image)

        cv2.imshow('Triple Jump Analysis', image)
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pose.close()

    # 7. Create and Save DataFrame
    if landmark_data:
        df = pd.DataFrame(landmark_data)
        df.to_csv(output_filename_csv, index=False)
        print(f"Successfully processed video. Data saved to {output_filename_csv}")
        print(f"Annotated video saved to {output_filename_video}")
    else:
        print("No data was extracted from the video.")
