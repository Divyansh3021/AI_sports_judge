import cv2
import mediapipe as mp
import csv
import numpy as np

def store_coordinates(file_path, coordinates):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame', 'Exercise', 'Body Part', 'X', 'Y', 'Z'])
        for frame_num, exercises in coordinates.items():
            for exercise, body_parts in exercises.items():
                for part_id, (x, y, z) in body_parts.items():
                    writer.writerow([frame_num, exercise, part_id, x, y, z])

def process_video(video_path, exercise):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Load video file
    cap = cv2.VideoCapture(video_path)

    # Initialize Mediapipe pose detection
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.6) as pose:
        frame_count = 0
        coordinates = {}

        while cap.isOpened():
            # Read the current frame
            success, image = cap.read()
            if not success:
                break

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            results = pose.process(image_rgb)

            # Store the coordinates of body parts
            body_parts = {}
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    body_parts[idx] = (landmark.x, landmark.y, landmark.z)

            # Save the body part coordinates for the current frame and exercise
            if frame_count in coordinates:
                coordinates[frame_count][exercise] = body_parts
            else:
                coordinates[frame_count] = {exercise: body_parts}

            # Draw the pose landmarks on the frame
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        return coordinates

pushups_video_path = 'pullup_test.mp4'
pushups_exercise = 'pullups'
coordinates = process_video(pushups_video_path, pushups_exercise)

store_coordinates('pull_up_test_motion_data.csv', coordinates)
