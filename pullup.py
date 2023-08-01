import cv2, mediapipe as mp
import math
import utils

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("perfect_pullup_2.mp4")
push_up = None
pull_up = None
body_alignment = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    prev_state = current_state = None
    
    count = 0
    elbow_shoulder_hip, shoulder_hip_ankle = utils.pullup_thresholds()
    states = ["s1", "s2", "s3"]

    while cap.isOpened():
        ret, frame = cap.read()
        try:
            frame_width, frame_height, _ = frame.shape

        except AttributeError:
            print("Attribute error occured")

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # print("Left: wrist ",left_wrist.y," shoulder", left_shoulder.y)
        # print("Right: wrist ",right_wrist.y," shoulder", right_shoulder.y)

        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # print("left wrist: ", left_wrist.x * frame_width, left_wrist.y * frame_height)
        # print("left shoulder: ", left_shoulder.x * frame_width, left_shoulder.y * frame_height)
        # print("left ankle: ", left_ankle)


        left_wrist.x, left_wrist.y  = left_wrist.x * frame_width, left_wrist.y * frame_height
        left_elbow.x, left_elbow.y  = left_elbow.x * frame_width, left_elbow.y * frame_height
        left_shoulder.x, left_shoulder.y = left_shoulder.x*frame_width, left_shoulder.y*frame_height
        left_hip.x, left_hip.y = left_hip.x*frame_width, left_hip.y*frame_height
        left_ankle.x, left_ankle.y = left_ankle.x*frame_width, left_ankle.y*frame_height

        right_wrist.x, right_wrist.y  = right_wrist.x * frame_width, right_wrist.y * frame_height
        right_elbow.x, right_elbow.y  = right_elbow.x * frame_width, right_elbow.y * frame_height
        right_shoulder.x, right_shoulder.y = right_shoulder.x*frame_width, right_shoulder.y*frame_height
        right_hip.x, right_hip.y = right_hip.x*frame_width, right_hip.y*frame_height
        right_ankle.x, right_ankle.y = right_ankle.x*frame_width, right_ankle.y*frame_height

        # print("Left wrist: ", left_wrist)

        #Calculating angles
        left_elbow_shoulder_hip_angle = utils.angle(left_elbow, left_wrist, left_shoulder)
        right_elbow_shoulder_hip_angle = utils.angle(right_elbow, right_shoulder, right_wrist)

        left_shoulder_hip_ankle_angle = utils.angle(left_hip, left_shoulder, left_ankle)
        right_shoulder_hip_ankle_angle = utils.angle(right_hip, right_shoulder, right_ankle)


        #checking alignment of person
        # body_angle = angle_of_singleline(left_shoulder, left_ankle)
        # print("Body angle is: ", body_angle)

        cv2.putText(frame, 'shoulder: Left: '+str(left_elbow_shoulder_hip_angle)+" Right: "+str(right_elbow_shoulder_hip_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'hip : Left: '+str(left_shoulder_hip_ankle_angle)+" Right: "+str(right_shoulder_hip_ankle_angle), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)        

        if ((elbow_shoulder_hip[0]>= left_elbow_shoulder_hip_angle>= elbow_shoulder_hip[1]) and (elbow_shoulder_hip[0]>= right_elbow_shoulder_hip_angle>= elbow_shoulder_hip[1])) and (prev_state == "s2"):
            current_state = states[0]

        elif ((elbow_shoulder_hip[0]>= left_elbow_shoulder_hip_angle>= elbow_shoulder_hip[1]) and (elbow_shoulder_hip[0]>= right_elbow_shoulder_hip_angle>= elbow_shoulder_hip[1])):
            # current_state = states[0]   #Current state = s1
            current_state = states[0]

        elif ((elbow_shoulder_hip[1] > left_elbow_shoulder_hip_angle >= elbow_shoulder_hip[2]) and (elbow_shoulder_hip[1] > right_elbow_shoulder_hip_angle >= elbow_shoulder_hip[2])) and (prev_state == "s1"):
            current_state = states[1]   #current state = s2

        
        if current_state:
            cv2.putText(frame, 'Current state: '+current_state, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if (prev_state == "s2" and current_state == "s1"):
            count+=1
            prev_state = None 
        
        prev_state = current_state

        cv2.putText(frame, "Count: "+str(count), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow("Mediapipe feed", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()