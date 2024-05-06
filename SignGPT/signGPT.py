import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QTextBrowser, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import mediapipe as mp
import pickle
import numpy as np
from chatGPT import respond


class HandDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.video_capture = cv2.VideoCapture(0)  # Access the default camera (index 0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        self.image_label = QLabel(self)
        self.text_edit = QTextBrowser(self)  # Create a QTextEdit widget
        self.text_edit1 = QPushButton("send")  # Create a QTextEdit widget
        self.textfield = QTextEdit(self)

        
        master_layout = QHBoxLayout()

        col1 = QVBoxLayout()
        col2 = QVBoxLayout ()


        col1.addWidget(self.text_edit, 50)  # Add the QTextEdit widget to the layout below the QLabel
        col1.addWidget(self.text_edit1)
        col2.addWidget(self.image_label, 70)
        col2.addWidget(self.textfield)


        master_layout.addLayout(col1, 40)
        master_layout.addLayout(col2, 60)

        self.setLayout(master_layout)

       
        self.text_edit1.clicked.connect(self.get_text) 
        self.text_edit1.clicked.connect(self.get_text) 
        
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()  # Update every 30 milliseconds

        self.setGeometry(100, 100, 800, 600) 

        self.sentence = []  # Define sentence list outside update_frame method
        self.letter_arr = ''
        self.final_letter = ['']
        self.output = ''

        self.text_edit1.clicked.connect(self.update_frame)
        # print(self.update_frame())


    def update_frame(self):

        def check_same_elements(arr):
            if len(arr) == 0:
                return True  # Empty array is considered to have all elements the same

            first_element = arr[0]
            for element in arr[1:]:
                if element != first_element:
                    return False  # Found an element different from the first element

            return True  # All elements are th




        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


        data_aux = []
        x_ = []
        y_ = []

        ret, frame = self.video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        H, W, _ = frame.shape

        results = self.hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            self.sentence.append(predicted_character)


            if len(self.sentence) > 10: 
                self.sentence = self.sentence[-10:]
                arr = self.sentence
                if check_same_elements(arr) == True:
                    self.letter_arr = arr[0]

                    current_text = self.textfield.toPlainText()
                    if len(current_text) == 0:
                            self.textfield.setText(self.letter_arr)
                    else:
                        if current_text[-1] != self.letter_arr:
                                update = current_text + self.letter_arr
                                self.textfield.setText(update)

            

            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(self.sentence), (3,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))
    
        return self.letter_arr

    def get_text(self):
        text = self.textfield.toPlainText()  # Retrieve text from QTextEdit widget
        self.result = respond(text)
        self.text_edit.setText(self.result)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hand_detection_app = HandDetectionApp()
    hand_detection_app.show()
    sys.exit(app.exec_())
