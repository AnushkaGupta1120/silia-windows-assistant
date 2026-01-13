import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# --- Configuration ---
wCam, hCam = 640, 480       # Camera resolution
frameR = 100                # Frame Reduction (padding from edges)
smoothening = 7             # Higher value = smoother but slower cursor

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,        # Only track one hand to avoid confusion
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Setup Screen & Variables ---
wScreen, hScreen = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pLocX, pLocY = 0, 0         # Previous Location (for smoothing)
cLocX, cLocY = 0, 0         # Current Location

def get_fingers_up(hand_landmarks):
    """Returns a list of 5 booleans (0 or 1) for each finger starting from Thumb."""
    fingers = []
    tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    # Thumb (Logic is different: x-axis comparison for left/right hand)
    # This logic assumes right hand facing camera
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers (y-axis comparison: tip above knuckle)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

print("Starting AI Mouse... Press 'q' to quit.")

while True:
    # 1. Capture Frame
    success, img = cap.read()
    if not success: break
    
    # Flip image horizontally for natural interaction (mirror view)
    img = cv2.flip(img, 1)
    
    # 2. Find Hand Landmarks
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw boundary box for "Active Area"
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of Index (8) and Middle (12) tips
            lmList = handLms.landmark
            x1, y1 = int(lmList[8].x * wCam), int(lmList[8].y * hCam)
            x2, y2 = int(lmList[12].x * wCam), int(lmList[12].y * hCam)
            
            # 3. Check which fingers are up
            fingers = get_fingers_up(handLms)
            # fingers format: [Thumb, Index, Middle, Ring, Pinky]

            # --- MODE 1: MOVING (Only Index Finger Up) ---
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert Coordinates (Interpolation)
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))
                
                # Smoothen Values
                cLocX = pLocX + (x3 - pLocX) / smoothening
                cLocY = pLocY + (y3 - pLocY) / smoothening
                
                # Move Mouse
                pyautogui.moveTo(cLocX, cLocY)
                pLocX, pLocY = cLocX, cLocY
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED) # Visual feedback

            # --- MODE 2: CLICKING (Index + Middle Up + Close together) ---
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                # Find distance between fingers
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                
                # Visual indicator for "Click Mode Ready"
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # If close enough, Click
                if length < 40:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.1) # Small delay to prevent double clicking instantly

            # --- MODE 3: SCROLL/SWIPE (Index + Middle + Ring Up) ---
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                # Using the position of the Middle finger to determine scroll direction
                # Scroll Up if hand is in top half, Down if in bottom half
                # Or simply: constant scroll
                pyautogui.scroll(-20) # Scroll down
                cv2.putText(img, "Scrolling...", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    # 4. Display
    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()