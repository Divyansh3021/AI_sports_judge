import cv2
import mediapipe as mp
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Function to store body part coordinates in a CSV file
def store_coordinates(file_path, coordinates):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame', 'Exercise', 'Body Part', 'X', 'Y', 'Z'])
        for frame_num, exercises in coordinates.items():
            for exercise, body_parts in exercises.items():
                for part_id, (x, y, z) in body_parts.items():
                    writer.writerow([frame_num, exercise, part_id, x, y, z])

# Function to process video frames, detect body parts, and store the data
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

# Function to train a random forest classifier
def train_classifier():
    # Load the motion data from the CSV file
    data = []
    labels = []
    with open('motion_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            frame_num, exercise, body_part, x, y, z = row
            data.append([float(x), float(y), float(z)])
            if exercise == "pullups":
                symbol = 0
            else:
                symbol = 1
            labels.append(symbol)

    # Encode the labels
    # label_encoder = LabelEncoder()
    # encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Define the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # Assuming 2 classes: pullups and pushups
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', test_acc)

    # Save the trained model
    model.save('trained_model.h5')


    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)

    # # Train a random forest classifier
    # # classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # classifier = KNeighborsClassifier(n_neighbors=20)
    # classifier.fit(X_train, y_train)

    # # Evaluate the classifier
    # y_pred = classifier.predict(X_test)
    # report = classification_report(y_test, y_pred)
    # print(report)

    # # Save the trained classifier
    # dump(classifier, 'trained_classifier.joblib')

    # return classifier

# Process pull-ups video
video_path = ['pullup_test.mp4','push_up-2.mp4',"pull_up-2.mp4","push_up-3.mp4","pull_up-3.mp4","push_up.mp4","pull_up.mp4","test-video-2.mp4"]
exercises = ['pullups',"pushups"]
coordinates_1 = {}
coordinates_2 = {}
coordinates_3 = {}
coordinates_4 = {}
coordinates_5 = {}
coordinates_6 = {}
coordinates_7 = {}
coordinates_8 = {}
flag = 1
for i in range(len(video_path)):
    if flag%2 == 0:
        exercise = exercises[1]
    else:
        exercise = exercises[0]
    flag+=1

    if i == 1:
        coordinates_1 = process_video(video_path[i], exercise)
    elif i == 2:
        coordinates_2 = process_video(video_path[i], exercise)
    elif i == 3:
        coordinates_3 = process_video(video_path[i], exercise)
    elif i == 4:
        coordinates_4 = process_video(video_path[i], exercise)
    elif i == 5:
        coordinates_5 = process_video(video_path[i], exercise)
    elif i == 6:
        coordinates_6 = process_video(video_path[i], exercise)
    elif i == 7:
        coordinates_7 = process_video(video_path[i], exercise)
    elif i == 8:
        coordinates_8 = process_video(video_path[i], exercise)

    print("processing {}th video".format(i))

    


# Process push-ups video
# pushups_video_path = 'test-video-2.mp4'
# pushups_exercise = 'pushups'
# pushups_coordinates = process_video(pushups_video_path, pushups_exercise)

combined_coordinates = {}
for d in [coordinates_1, coordinates_2, coordinates_3, coordinates_4, coordinates_5, coordinates_6, coordinates_7, coordinates_8]:
    print("\n\n\n\n\n\n")
    print(d)
    combined_coordinates.update(d)
# Combine the coordinates from both videos
# combined_coordinates = {**coordinates_1, **coordinates_2, **coordinates_3, **coordinates_4, **coordinates_5, **coordinates_6, **coordinates_7, **coordinates_8}

# Save the body part coordinates to a CSV file
# store_coordinates('motion_data.csv', combined_coordinates)

# Train the classifier
# train_classifier()
