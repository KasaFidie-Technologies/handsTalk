import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QTextEdit, QPushButton, QGridLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
import cv2
import mediapipe as mp
import pickle
import numpy as np
from chatGPT import respond


class HandDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('signGPT')
        
        
        self.video_capture = cv2.VideoCapture(0)  # Access the default camera (index 0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        layout = QGridLayout()
        self.setLayout(layout)

        self.camera_display = QLabel()
        self.camera_display.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("background-color: lightgray; border: 2px solid gray;")
        self.text_display_sign = QTextEdit()
        self.text_display_sign.setReadOnly(True)
        self.text_display_sign.setStyleSheet("background-color: lightgray; border: 2px solid gray;")
        self.submit_button = QPushButton('Submit')
        self.submit_button.setStyleSheet("background-color: #007BFF; color: white; border: 2px solid #007BFF;")
        self.submit_button.setFixedHeight(40)  # Set the height of the button
        self.textfield = QTextEdit()
        self.textfield.setStyleSheet("background-color: dark-grey; border: 2px solid gray; color: white")


        # placeholders
        self.set_placeholder_text(self.text_display_sign, 'sign text here...')
        self.set_placeholder_text(self.text_display, 'chatGPT text here...')
        self.set_placeholder_text(self.textfield, 'converted sign here...')

        font = QFont()
        font.setPointSize(14)
        self.text_display.setFont(font)
        self.text_display_sign.setFont(font)


        self.camera_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_display_sign.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        layout.addWidget(self.text_display, 0, 0, 2, 1)
        layout.addWidget(self.text_display_sign, 2, 0)
        layout.addWidget(self.submit_button, 3, 0)
        layout.addWidget(self.camera_display, 0, 1, 4, 1)
        layout.addWidget(self.textfield, 0, 1, 2, 1)

        #  connect when clicked  
        self.submit_button.clicked.connect(self.get_text) 
        # self.text_edit1.clicked.connect(self.get_text)
        self.submit_button.clicked.connect(self.copy_images_for_letters)
        # self.submit_button.clicked.connect(self.update_frame)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()  # Update every 30 milliseconds

        self.setGeometry(400, 400, 1000, 600) 

        self.sentence = []  # Define sentence list outside update_frame method
        self.letter_arr = ''
        self.final_letter = ['']
        self.output = ''

    def set_placeholder_text(self, text_edit, placeholder_text):
        text_edit.setPlaceholderText(placeholder_text)

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

        labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
                       11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
                       21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


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

            

            cv2.rectangle(frame, (0,0), (640, 40), (45, 17, 16), -1)
            cv2.putText(frame, ' '.join(self.sentence), (3,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.camera_display.setPixmap(QPixmap.fromImage(image))
    
        return self.letter_arr

    def get_text(self):
        text = self.textfield.toPlainText()  # Retrieve text from QTextEdit widget
        self.result = respond(text)
        self.text_display.setText(self.result)

        return self.result

    def copy_images_for_letters(self):
        letters = self.get_text()

        letter = letters.lower()
        letters = [char for char in letter]
        signAlpha = {1:'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
                     11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
                     21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: ' ', 28: '.', 29: ',',
                     30: '0', 31: '1', 32: '2', 33: '3', 34: '4', 35: '5', 36: '6', 37: '7', 38: '8', 39: '9'}
        symbols = {1: '😍', 2: '👈', 3: '🤘', 4: '🤚', 5: '🤙', 6: '✋', 7: '🤛', 8: '🖐', 9: '🤞',
                   10: '✊', 11: ' 👊', 12: '👍', 13: '👆', 14: '👇', 15: '👌', 16: '👎', 17: '🤝', 18: '😨',
                   19: '🖖', 20: '👋', 21: '🙌', 22: '🤜', 23: '👐', 24: '👉', 25: '👏', 26: '🙏', 27: '   ', 
                   28: '.', 29: ',',
                   30: '0', 31: '1', 32: '2', 33: '3', 34: '4', 35: '5', 36: '6', 37: '7', 38: '8', 39: '9'}

        final_text = []
        

        for letter in letters:
            for i in signAlpha:
                if letter == signAlpha[i]:
                    result_sign = symbols[i]
                    final_text.append(result_sign)

        final_text = " ".join(final_text)
        self.text_display_sign.setText(final_text)
        final_text = []
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hand_detection_app = HandDetectionApp()
    hand_detection_app.show()
    sys.exit(app.exec_())
