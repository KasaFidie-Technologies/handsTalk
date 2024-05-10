# Hand Detection and Sign Language Conversion App

This PyQt5 application utilizes hand detection algorithm to convert sign language symbols into plain text. The application captures video from the camera, processes it using MediaPipe for hand detection, and converts the detected sign language symbols into text, the texts are sent to chatGPT 3.5 turbo for response and the answer recieved is diplayed on the output.

## Features
- Real-time hand detection and sign language symbol recognition
- Conversion of language symbol into to text
- Display of text corresponding to the detected sign language symbols
- Text input field for composing messages using sign language symbols
- Text display field for viewing the composed messages


## Requirements
All requirement are stated in the requirement.txt file

## Installation
1. Clone the repository: `git clone git@github.com:KasaFidie-Technologies/handsTalk.git`
2. On `main` branch cd **SignGPT** directory
3. Install the required dependencies: `pip3 install -r requirements.txt`
4. Run project: `python signGPT.py`


## Usage
To use this app first check the image file to know the possible sign detections images.
1. Run the application by executing the Python script.
2. Position your hand in front of the camera to start detecting gestures.
3. As you make hand gestures, the application will convert them into textss and display the corresponding sign language symbols.
4. Compose messages using the sign language symbols in the text input field.
5. View the composed messages in the text display field.
6. Click on submit button after composing your message for response from chatGPT 3.5 turbo
7. You can also edit message using the keyboard


## Contributor
- Ezra Asiedu
[https://github.com/Kwasi633]

- Aliu Tijani
[https://github.com/Aliu2211]

- Enoch Djam Beamahn
[https://github.com/djamkenny]

## Acknowledgements
- Thanks to the developers of PyQt5, OpenCV, MediaPipe, and especially openAI for providing the tools that enabled this project.
-  Thanks to ATF Challenge who has place us on enthusiastic development atmosphere.

## video demo
[https://drive.google.com/file/d/12NCj1gQI6q7Gu56RuQ1pHcAuvn0Metdr/view?usp=sharing]


## Bigger Vision

This app in the future should be able to improve all signer's communication with AI tools and in all sectors. 
