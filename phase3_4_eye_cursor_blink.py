import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import os
import subprocess
import threading
import speech_recognition as sr
import time
import webbrowser

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= SYSTEM MODE =================
MODE = "SLEEP"   # SLEEP | VOICE | EYE

# ================= SCREEN SETUP =================
pyautogui.FAILSAFE = False

# ================= CAMERA SETUP =================
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# ================= FACE LANDMARKER =================
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ================= LANDMARK IDS =================
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# ================= EYE CONTROL TUNING =================
DEAD_ZONE = 0.0015
SENSITIVITY = 3500
VELOCITY_SMOOTHING = 2
vel_x, vel_y = 0, 0

# ================= BLINK CONTROL =================
EAR_THRESHOLD = 0.19
BLINK_FRAMES = 2
CLICK_COOLDOWN = 1.0
blink_counter = 0
last_click_time = 0

# ================= DANGEROUS COMMAND CONFIRM =================
pending_dangerous_command = None

print("System started. Say 'hello' to activate voice mode.")

# ================= HELPER FUNCTIONS =================
def get_normalized_iris_offset(face, iris_ids, eye_ids):
    iris_x = sum(face[i].x for i in iris_ids) / len(iris_ids)
    iris_y = sum(face[i].y for i in iris_ids) / len(iris_ids)
    eye_x = sum(face[i].x for i in eye_ids) / len(eye_ids)
    eye_y = sum(face[i].y for i in eye_ids) / len(eye_ids)
    return iris_x - eye_x, iris_y - eye_y


def eye_aspect_ratio(face, eye_ids):
    v1 = np.linalg.norm(
        np.array([face[eye_ids[1]].x, face[eye_ids[1]].y]) -
        np.array([face[eye_ids[5]].x, face[eye_ids[5]].y])
    )
    v2 = np.linalg.norm(
        np.array([face[eye_ids[2]].x, face[eye_ids[2]].y]) -
        np.array([face[eye_ids[4]].x, face[eye_ids[4]].y])
    )
    h = np.linalg.norm(
        np.array([face[eye_ids[0]].x, face[eye_ids[0]].y]) -
        np.array([face[eye_ids[3]].x, face[eye_ids[3]].y])
    )
    return (v1 + v2) / (2.0 * h)


def open_any_app(app_name):
    try:
        subprocess.Popen(app_name, shell=True)
    except:
        print("Could not open app:", app_name)


def execute_system_command(command):
    global pending_dangerous_command

    command = command.lower()

    # -------- SEARCH --------
    if command.startswith("search for"):
        query = command.replace("search for", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")

    # -------- OPEN COMMON APPS --------
    elif "notepad" in command:
        open_any_app("notepad")

    elif "calculator" in command or "calc" in command:
        open_any_app("calc")

    elif "vscode" in command or "visual studio code" in command:
        open_any_app("code")

    elif "chrome" in command:
        open_any_app("chrome")

    # -------- OPEN SYSTEM LOCATIONS --------
    elif "file explorer" in command or "explorer" in command:
        os.startfile("explorer")

    elif "downloads" in command:
        os.startfile(os.path.join(os.path.expanduser("~"), "Downloads"))

    elif "documents" in command:
        os.startfile(os.path.join(os.path.expanduser("~"), "Documents"))

    elif "desktop" in command:
        os.startfile(os.path.join(os.path.expanduser("~"), "Desktop"))

    elif "recycle bin" in command:
        subprocess.Popen(["explorer", "shell:RecycleBinFolder"], shell=True)

    # -------- CLOSE --------
    elif "close" in command:
        pyautogui.hotkey("alt", "f4")

    # -------- DANGEROUS --------
    elif "shutdown" in command:
        pending_dangerous_command = "shutdown"
        print("Say YES to confirm shutdown")

    elif "restart" in command:
        pending_dangerous_command = "restart"
        print("Say YES to confirm restart")

    else:
        print("Unknown command:", command)


# ================= VOICE LISTENER =================
def voice_listener():
    global MODE, pending_dangerous_command

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=4)

            text = recognizer.recognize_google(audio).lower()
            print("Heard:", text)

            # ---- CONFIRMATION ----
            if pending_dangerous_command:
                if "yes" in text:
                    if pending_dangerous_command == "shutdown":
                        os.system("shutdown /s /t 5")
                    elif pending_dangerous_command == "restart":
                        os.system("shutdown /r /t 5")
                pending_dangerous_command = None
                continue

            # ---- MODE CONTROL ----
            if MODE == "SLEEP" and "hello" in text:
                MODE = "VOICE"
                print("VOICE mode activated")

            elif MODE == "VOICE":
                if "use eyes" in text or "eye control" in text:
                    MODE = "EYE"
                    print("EYE mode activated")

                elif "stop" in text or "sleep" in text:
                    MODE = "SLEEP"
                    print("System sleeping")

                else:
                    execute_system_command(text)

        except:
            pass


threading.Thread(target=voice_listener, daemon=True).start()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # -------- MODE CHECK --------
    if MODE != "EYE":
        cv2.putText(frame, f"MODE: {MODE}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SILIA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # -------- EYE MODE --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:
        face = result.face_landmarks[0]

        dx_l, dy_l = get_normalized_iris_offset(face, LEFT_IRIS, LEFT_EYE)
        dx_r, dy_r = get_normalized_iris_offset(face, RIGHT_IRIS, RIGHT_EYE)

        dx = (dx_l + dx_r) / 2
        dy = (dy_l + dy_r) / 2

        if abs(dx) < DEAD_ZONE:
            dx = 0
        if abs(dy) < DEAD_ZONE:
            dy = 0

        target_vx = dx * SENSITIVITY
        target_vy = dy * SENSITIVITY

        vel_x += (target_vx - vel_x) / VELOCITY_SMOOTHING
        vel_y += (target_vy - vel_y) / VELOCITY_SMOOTHING

        pyautogui.moveRel(vel_x, vel_y)

        # ---- BLINK CLICK ----
        left_ear = eye_aspect_ratio(face, LEFT_EYE_EAR)
        right_ear = eye_aspect_ratio(face, RIGHT_EYE_EAR)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                if time.time() - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = time.time()
            blink_counter = 0

    cv2.putText(frame, "MODE: EYE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("SILIA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
