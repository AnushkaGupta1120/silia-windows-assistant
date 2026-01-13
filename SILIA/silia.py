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
import pyttsx3

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= SYSTEM MODE =================
MODE = "SLEEP"   # SLEEP | VOICE | EYE

# ================= COMMAND DEBOUNCE =================
last_voice_command = ""
last_command_time = 0
COMMAND_COOLDOWN = 1.5  # seconds

# ================= TEXT TO SPEECH =================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 175)
tts_engine.setProperty("volume", 1.0)

def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except:
        pass

# ================= CAMERA SETUP =================
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pyautogui.FAILSAFE = False

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

# ================= EYE CONTROL =================
DEAD_ZONE = 0.0015
SENSITIVITY = 3500
SMOOTHING = 2
vel_x, vel_y = 0, 0

# ================= BLINK CONTROL =================
EAR_THRESHOLD = 0.19
BLINK_FRAMES = 2
CLICK_COOLDOWN = 1.0
blink_counter = 0
last_click_time = 0

# ================= DANGEROUS CONFIRM =================
pending_dangerous_command = None

print("SILIA started. Say 'hey windows' to activate.")

# ================= HELPER FUNCTIONS =================
def iris_offset(face, iris_ids, eye_ids):
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


# ================= APP OPENERS (WINDOWS SAFE) =================
def open_chrome():
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            subprocess.Popen(path)
            return
    speak("Chrome not found")


def open_notepad():
    subprocess.Popen("notepad.exe")


def open_calculator():
    subprocess.Popen("calc.exe")


# ================= COMMAND EXECUTOR =================
def execute_command(cmd):
    global pending_dangerous_command
    global last_voice_command, last_command_time

    now = time.time()
    if cmd == last_voice_command and now - last_command_time < COMMAND_COOLDOWN:
        return

    last_voice_command = cmd
    last_command_time = now

    cmd = cmd.lower()

    # ---- SEARCH ----
    if cmd.startswith("search for"):
        query = cmd.replace("search for", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        speak("Searching")

    # ---- APPS ----
    elif "chrome" in cmd:
        open_chrome()
        speak("Opening Chrome")

    elif "notepad" in cmd:
        open_notepad()
        speak("Opening Notepad")

    elif "calculator" in cmd or "calc" in cmd:
        open_calculator()
        speak("Opening Calculator")

    # ---- SYSTEM LOCATIONS ----
    elif "file explorer" in cmd or "explorer" in cmd:
        os.startfile("explorer")
        speak("Opening File Explorer")

    elif "downloads" in cmd:
        os.startfile(os.path.join(os.path.expanduser("~"), "Downloads"))
        speak("Opening Downloads")

    elif "desktop" in cmd:
        os.startfile(os.path.join(os.path.expanduser("~"), "Desktop"))
        speak("Opening Desktop")

    elif "documents" in cmd:
        os.startfile(os.path.join(os.path.expanduser("~"), "Documents"))
        speak("Opening Documents")

    elif "recycle bin" in cmd:
        subprocess.Popen(["explorer", "shell:RecycleBinFolder"], shell=True)
        speak("Opening Recycle Bin")

    # ---- CLOSE ----
    elif "close" in cmd:
        pyautogui.hotkey("alt", "f4")
        speak("Closing window")

    # ---- DANGEROUS ----
    elif "shutdown" in cmd:
        pending_dangerous_command = "shutdown"
        speak("Are you sure? Say yes to confirm")

    elif "restart" in cmd:
        pending_dangerous_command = "restart"
        speak("Are you sure? Say yes to confirm")


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

            # ---- CONFIRM DANGEROUS ----
            if pending_dangerous_command:
                if "yes" in text:
                    if pending_dangerous_command == "shutdown":
                        os.system("shutdown /s /t 5")
                    elif pending_dangerous_command == "restart":
                        os.system("shutdown /r /t 5")
                pending_dangerous_command = None
                continue

            # ---- MODE CONTROL ----
            if MODE == "SLEEP" and "hey windows" in text:
                MODE = "VOICE"
                speak("Yes, I'm listening")

            elif MODE == "VOICE":
                if "use eyes" in text or "eye control" in text:
                    MODE = "EYE"
                    speak("Eye control activated")

                elif "stop" in text or "sleep" in text:
                    MODE = "SLEEP"
                    speak("Okay, sleeping")

                else:
                    execute_command(text)

        except:
            pass


threading.Thread(target=voice_listener, daemon=True).start()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ---- CAMERA PAUSED IN VOICE/SLEEP ----
    if MODE != "EYE":
        cv2.putText(frame, f"MODE: {MODE}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SILIA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ---- EYE MODE ----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:
        face = result.face_landmarks[0]

        dx_l, dy_l = iris_offset(face, LEFT_IRIS, LEFT_EYE)
        dx_r, dy_r = iris_offset(face, RIGHT_IRIS, RIGHT_EYE)

        dx = (dx_l + dx_r) / 2
        dy = (dy_l + dy_r) / 2

        if abs(dx) < DEAD_ZONE:
            dx = 0
        if abs(dy) < DEAD_ZONE:
            dy = 0

        vel_x += (dx * SENSITIVITY - vel_x) / SMOOTHING
        vel_y += (dy * SENSITIVITY - vel_y) / SMOOTHING

        pyautogui.moveRel(vel_x, vel_y)

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
