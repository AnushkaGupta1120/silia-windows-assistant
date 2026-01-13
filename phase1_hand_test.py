import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load hand model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Phase 1: Hand Tracking Test (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Phase 1 - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
