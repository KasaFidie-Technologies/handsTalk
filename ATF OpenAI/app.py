# APP
import cv2 
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from scipy import stats
from keras.models import Sequential
from keras.layers import LSTM, Dense

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 


# Fetch for the data in PATH

DATA_PATH = os.path.join('MP_Data')

# Actions
actions = np.array(['What', 'is'])


# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(sequence_length):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))



def mediapipe_detection(image, model):
     
    # COLOR COVERSION from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SET THE IMAGE TO NON WRITABLE
    image.flags.writeable = False
    #  MAKE PREDICTION 
    results = model.process(image)
    # COLOR CONVERSION from  RGB to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # return image and result
    return image, results


def draw_landmarks(image, results):
    # Draw face connection 
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS)
    # draw pose connection 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    #  draw left connectio 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    #  draw right connection 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
     # Draw face connection 
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 50, 30), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(80, 150, 130), thickness=1, circle_radius=1)
                              )
    # draw pose connection 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(60, 30, 30), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(70, 180, 230), thickness=1, circle_radius=1)
                               )

    #  draw left connectio 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 40, 80), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(100, 50, 130), thickness=1, circle_radius=1)
                               )

    #  draw right connection 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 50, 10), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(180, 50, 190), thickness=1, circle_radius=1))

# Extract Kepoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


plt.imshow(prob_viz(res, actions, image, colors))

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(2,keypoints)
        sequence = sequence[:30]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res)
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        # #3. Viz logic
            if np.unique(predictions[-1:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                        
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
        # Show to screen
        image = cv2.resize(image, (1250, 900))
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully*
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

