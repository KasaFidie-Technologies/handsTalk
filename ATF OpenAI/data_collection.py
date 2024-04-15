# THIS FILE COLLECT DATA AND STORED IT IN ANOTHER FILE FOR PROCESSING

# Importing some dependencies
import cv2, os, time
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp

# Import drawing_utils and drawing_styles.
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 


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
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACED_CONNECTIONS,
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


def resize(frame):
    return cv2.resize

# |Storing the data|

# Path to export the data
DATA_PATH = os.path.join('MP_Data') # currently dir and create a dir


# <files that store the action to be detected>
actions = np.array(['What', 'is'])

# <thirty videos of data files>
no_sequences = 30

# <10 frames videos>
sequence_length = 30

# Loop that mkdir 

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# |Collecting the data|

cap = cv2.VideoCapture(0)

# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # make detections
                image, results = mediapipe_detection(frame, holistic)

                # draw landmarks
                draw_styled_landmarks(image, results)

                # Waiting logic

                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    image = cv2.resize(image, (1250, 900))
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Show to screen
                    image = cv2.resize(image, (1250, 900))
                    cv2.imshow('OpenCV Feed', image)

                # Eporting keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)


                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()