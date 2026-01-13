import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Load Face Mesh Model ----------
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Phase 2: Face Mesh + Iris Test (Press Q to quit)")

# Iris landmark indices (MediaPipe standard)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

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

    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:
        for face in result.face_landmarks:

            # Draw all face landmarks (light dots)
            for lm in face:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Draw iris landmarks (larger dots)
            for idx in LEFT_IRIS + RIGHT_IRIS:
                lm = face[idx]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("Phase 2 - Face Mesh & Iris", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
