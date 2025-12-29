FaceWave-AI


A small experiment in controlling computers with hands, faces & expressions
FaceWave-AI is a personal exploration into vision-based interaction.
It’s a simple real-time prototype where your camera becomes the controller — your hand gestures, face, and even basic emotion cues are read and turned into live feedback on screen.
so ill explain what it does.....


Detects hand gestures and counts visible fingers
Shows live overlay feedback so you always know what it’s reading


Supports simple emotion states (like happy / neutral / surprised)
Can remember faces locally, so the system can greet you next time
Everything runs locally on your machine.


🎯 Why I built it
Mostly curiosity.
I’ve always liked the idea of interaction that feels a bit more human and playful — especially useful for:

experimental game prototype
accessibility concept

This project is my attempt at learning and building something that feels alive, not just functional.


🖐️ Gestures
The prototype understands:
0 fingers → idle/stop
1 finger → select
2 fingers → wave
3 fingers → action
5 fingers → open palm
They don’t trigger anything crazy (this is a base tech demo), but they show how a gesture system could be used.


(Face Memory)
Press R while your face is visible
Type your name
Next time the program runs, it remembers you 
Stored locally. Not uploaded anywhere.


 How to Run????
Install dependencies:

pip install -r requirements.txt

Run:

python3 gesture.py

Requires a webcam.


🔐 About privacy

No cloud.
No uploads.
No hidden tracking.
Just local files.


👤 Made by
Ishpreet Singh Chouhan
University of Alberta — CS (AI specialization)
Built mostly for learning, fun, and curiosity.
