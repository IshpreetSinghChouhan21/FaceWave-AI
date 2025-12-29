import cv2
import mediapipe as mp
import urllib.request
import os
import numpy as np
import time


##################################################
# DOWNLOAD HAND MODEL
##################################################
HAND_MODEL = "hand_landmarker.task"

if not os.path.exists(HAND_MODEL):
    print("Downloading hand model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL
    )

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode


##################################################
# HAND LANDMARKER
##################################################
HandOptions = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


##################################################
# FACE MESH
##################################################
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


##################################################
# PERSISTENT FACE STORAGE
##################################################
FACE_DB = "faces.npy"
known_faces = {}  # name -> embedding vector


def save_faces():
    if not known_faces:
        return
    np.save(FACE_DB, known_faces)
    print("Saved faces to disk.")


def load_faces():
    global known_faces
    if os.path.exists(FACE_DB):
        try:
            known_faces = np.load(FACE_DB, allow_pickle=True).item()
            print(f"Loaded {len(known_faces)} faces from disk.")
        except:
            print("Face database corrupted — resetting.")
            known_faces = {}


##################################################
# FACE EMBEDDINGS
##################################################
def landmarks_to_embedding(face_landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
    pts = pts - pts.mean(axis=0)
    norm = np.linalg.norm(pts)
    if norm > 0:
        pts /= norm
    return pts.flatten()


def recognize_face(current_embedding):
    if not known_faces:
        return None

    best_name = None
    best_score = 10

    for name, emb in known_faces.items():
        dist = np.linalg.norm(current_embedding - emb)
        if dist < best_score:
            best_score = dist
            best_name = name

    # STRICTER threshold (more accuracy)
    return best_name if best_score < 0.95 else None


##################################################
# FINGER COUNT
##################################################
def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(1 if landmarks[tips[0]].x < landmarks[tips[0] - 1].x else 0)

    for i in range(1, 5):
        fingers.append(1 if landmarks[tips[i]].y < landmarks[tips[i] - 2].y else 0)

    return sum(fingers)


##################################################
# EMOTION
##################################################
def get_emotion(face_landmarks, frame):
    h, w, _ = frame.shape
    lm = face_landmarks.landmark

    left = lm[61]
    right = lm[291]
    top = lm[13]
    bottom = lm[14]

    mouth_width = abs((right.x - left.x) * w)
    mouth_height = abs((bottom.y - top.y) * h)

    ratio = mouth_width / mouth_height if mouth_height != 0 else 0

    if ratio > 2.0:
        return "Happy 😄"
    elif ratio > 1.6:
        return "Neutral 🙂"
    else:
        return "Surprised 😯"


##################################################
# GESTURE HUD TEXT
##################################################
def gesture_action(fingers):
    if fingers == 0:
        return "Command: STOP ✋"
    if fingers == 1:
        return "Command: SELECT 👍"
    if fingers == 2:
        return "Command: WAVE ✌️"
    if fingers == 3:
        return "Command: ACTION 🤟"
    if fingers == 5:
        return "Command: OPEN PALM 🖐️"
    return "Command: Idle"


##################################################
# FACE RECOG STABILITY FILTER
##################################################
last_name = None
stable_name = None
stable_count = 0


##################################################
# CAMERA
##################################################
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

with vision.HandLandmarker.create_from_options(HandOptions) as hand_model:

    load_faces()
    print("Gesture + Emotion + Face Recognition Running — Press Q to quit.")
    print("Press R to register your face with your name.")

    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ##################################################
        # HAND
        ##################################################
        hand_result = hand_model.detect_for_video(mp_image, timestamp)

        command = "Command: Idle"

        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            fingers = count_fingers(hand)
            command = gesture_action(fingers)

            cv2.putText(frame, f"Fingers: {fingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            for lm in hand:
                h,w,_ = frame.shape
                x,y = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (x,y), 5, (0,255,0), -1)


        ##################################################
        # FACE + EMOTION + RECOGNITION
        ##################################################
        face = face_mesh.process(rgb)

        name_display = "No Face"

        if face.multi_face_landmarks:
            fl = face.multi_face_landmarks[0]

            emotion = get_emotion(fl, frame)

            emb = landmarks_to_embedding(fl)
            who = recognize_face(emb)

            

            if who:
                if who == last_name:
                    stable_count += 1
                else:
                    stable_count = 0

                last_name = who

                # must be stable for a few frames
                if stable_count > 6:
                    stable_name = who

            if stable_name:
                name_display = f"Hi {stable_name} 😎"
            else:
                name_display = "Unknown Face"

            cv2.putText(frame, f"Emotion: {emotion}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        ##################################################
        # HUD
        ##################################################
        cv2.putText(frame, command, (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)

        cv2.putText(frame, name_display, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 3)

        cv2.putText(frame, "Press R to register your face", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        ##################################################
        # SHOW
        ##################################################
        cv2.imshow("AI Gesture + Emotion + Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        ##################################################
        # REGISTER NEW FACE
        ##################################################
        if key == ord('r') and face.multi_face_landmarks:
            print("Enter name in terminal (e.g., Ishpreet): ")
            user_name = input("> ").strip()
            if user_name:
                emb = landmarks_to_embedding(face.multi_face_landmarks[0])
                known_faces[user_name] = emb
                save_faces()
                print(f"Saved face for {user_name}")

        timestamp += 33

cap.release()
cv2.destroyAllWindows()















