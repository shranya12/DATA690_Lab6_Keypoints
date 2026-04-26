import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    status = "NORMAL"

    # Keypoints detection
    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = result.pose_landmarks.landmark

        nose_y = lm[0].y
        left_hand_y = lm[15].y
        right_hand_y = lm[16].y

        # 🚨 Anomaly rule: hands above head
        if left_hand_y < nose_y or right_hand_y < nose_y:
            status = "ANOMALY DETECTED"

    else:
        status = "NO PERSON"

    # 🎨 Iron Man-style UI
    cv2.putText(frame, "IRON MAN HUD SYSTEM", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"STATUS: {status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if status != "NORMAL" else (0, 255, 0), 2)

    cv2.putText(frame, "SCANNING ENVIRONMENT...", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("DATA690 Keypoints Lab", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
