import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# --- Configuration ---
wCam, hCam = 1280, 720      # Increased resolution for better keyboard space
frameR = 100                # Frame Reduction
smoothening = 7
click_threshold = 40        # Distance to trigger click

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Setup Screen ---
wScreen, hScreen = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# --- Variables ---
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0
keyboard_active = False     # Toggle for keyboard mode
last_click_time = 0         # To prevent spamming keys

# --- Keyboard Class ---
class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

# Create Keyboard Layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

buttonList = []
start_x, start_y = 50, 100  # Starting position of keyboard
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([start_x + j * 100, start_y + i * 100], key))

def draw_keyboard(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # Draw key background
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        # Draw text
        cv2.putText(img, button.text, (x + 20, y + 65), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

def get_fingers_up(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # Thumb (Right hand assumption)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
        
    # Other 4 fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

print("Running Super Mouse...")
print("Gestures: Index=Move | Pinch=Click | Rock'n'Roll(Index+Pinky)=Right Click")
print("Press 'k' to toggle Keyboard Mode.")
print("Press 'q' to Quit.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw Active Area Box
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # --- LOGIC START ---
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            
            lmList = handLms.landmark
            # Index Finger Tip (8) Coordinates
            x1, y1 = int(lmList[8].x * wCam), int(lmList[8].y * hCam)
            # Middle Finger Tip (12) Coordinates
            x2, y2 = int(lmList[12].x * wCam), int(lmList[12].y * hCam)
            
            fingers = get_fingers_up(handLms)
            # fingers = [Thumb, Index, Middle, Ring, Pinky]

            # Calculate Distance for Clicking (Index & Middle)
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # =================================================
            # MODE A: VIRTUAL KEYBOARD (Active if 'k' was pressed)
            # =================================================
            if keyboard_active:
                img = draw_keyboard(img, buttonList)
                
                # Check if Index finger is over any key
                for button in buttonList:
                    bx, by = button.pos
                    bw, bh = button.size
                    
                    if bx < x1 < bx + bw and by < y1 < by + bh:
                        # Highlight button when hovering
                        cv2.rectangle(img, button.pos, (bx + bw, by + bh), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (bx + 20, by + 65), 
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        
                        # Check for Click (Pinch)
                        if distance < click_threshold and (time.time() - last_click_time > 0.3):
                            cv2.rectangle(img, button.pos, (bx + bw, by + bh), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, button.text, (bx + 20, by + 65), 
                                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            
                            # Type the letter!
                            pyautogui.press(button.text)
                            last_click_time = time.time() # Reset timer
            
            # =================================================
            # MODE B: MOUSE CONTROLLER (Active if Keyboard is OFF)
            # =================================================
            else:
                # 1. MOVEMENT (Index up, Middle down)
                if fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 0:
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))
                    
                    cLocX = pLocX + (x3 - pLocX) / smoothening
                    cLocY = pLocY + (y3 - pLocY) / smoothening
                    
                    pyautogui.moveTo(cLocX, cLocY)
                    pLocX, pLocY = cLocX, cLocY
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                # 2. LEFT CLICK (Index + Middle Up + Pinch)
                if fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 0:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    if distance < click_threshold:
                        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()
                        time.sleep(0.15)

                # 3. RIGHT CLICK (Index + Pinky Up "Rock n Roll")
                if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0:
                    cv2.putText(img, "Right Click", (x1, y1 - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    pyautogui.rightClick()
                    time.sleep(0.3) # Delay to prevent spamming

    # Display
    cv2.imshow("Super AI Mouse", img)
    
    # Key Controls
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Quit
        break
    if key & 0xFF == ord('k'):  # Toggle Keyboard
        keyboard_active = not keyboard_active

cap.release()
cv2.destroyAllWindows()