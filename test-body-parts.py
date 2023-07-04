import csv
import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras

# Function to load motion data from a CSV file
def load_motion_data(file_path):
    exercise = []
    data = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            frame_num, label, body_part, x, y, z = row
            exercise.append(label)
            data.append([float(x), float(y), float(z)])
    return exercise, data

# Function to classify motion data using the trained classifier
def classify_motion(data, classifier):
    predictions = classifier.predict(data)
    return predictions

# Load the trained classifier
model = keras.models.load_model('trained_model.h5')

# Load the motion data from the CSV file
motion_data_file = 'pull_up_test_motion_data.csv'
labels, motion_data = load_motion_data(motion_data_file)

# Classify the motion data
predictions = classify_motion(motion_data, model)
predicted_classes = np.argmax(predictions, axis=1)
# report = classification_report(labels, predictions)

# Print the predictions
total_len = len(predicted_classes)
pullups = 0
pushups = 0
for i in range(total_len):
    if i == 0:
        pullups+=1
    else:
        pushups+=1

print("Pushup Prob:", pushups/total_len)
print("Pullup Prob:", pullups/total_len)
