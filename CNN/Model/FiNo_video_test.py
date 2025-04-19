import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnxruntime as ort
import time
from retinaface import RetinaFace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ONNX modellek betöltése
eye_session = ort.InferenceSession("eye_model.onnx")
mouth_session = ort.InferenceSession("mouth_model.onnx")

# Transzformációk
transform_eye = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_mouth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Paraméterek
window_duration = 30
closed_threshold = 0.4
continuous_closed_threshold = 2.0

total_closed_time = 0.0
window_start_time = time.time()
eye_closed_continuous_time = 0.0
prev_time = time.time()

frame_count = 0
landmarks_last = None

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time_fps = time.time()
    frame_count += 1

    # Feldolgozási felbontás csökkentése
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    overall_label = "Monitoring"
    text_color = (255, 255, 255)
    prob_left, prob_right = 0.0, 0.0
    mouth_label = "Unknown"
    frame_closed = False

    # Arc detektálás
    if frame_count % 3 == 0 or landmarks_last is None:
        detections = RetinaFace.detect_faces(small_rgb)
        if isinstance(detections, dict) and len(detections) > 0:
            face = list(detections.values())[0]
            landmarks_last = face["landmarks"]

    landmarks = landmarks_last

    if landmarks:
        # Skálázás vissza teljes méretre
        def scale_point(pt):
            return (int(pt[0] / scale), int(pt[1] / scale))

        left_eye = scale_point(landmarks["left_eye"])
        right_eye = scale_point(landmarks["right_eye"])
        mouth_left = scale_point(landmarks["mouth_left"])
        mouth_right = scale_point(landmarks["mouth_right"])

        eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        roi_size = int(eye_dist * 0.7)
        roi_size = max(20, min(roi_size, 100))

        def get_eye_roi(center):
            x1 = int(center[0] - roi_size // 2)
            y1 = int(center[1] - roi_size // 2)
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            return frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]

        roi_left = get_eye_roi(left_eye)
        roi_right = get_eye_roi(right_eye)

        # Szem predikció
        if roi_left.size and roi_right.size:
            input_left = transform_eye(Image.fromarray(cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY))).unsqueeze(0).numpy()
            input_right = transform_eye(Image.fromarray(cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY))).unsqueeze(0).numpy()

            output_left = eye_session.run(None, {eye_session.get_inputs()[0].name: input_left})[0]
            output_right = eye_session.run(None, {eye_session.get_inputs()[0].name: input_right})[0]

            prob_left = 1 / (1 + np.exp(-output_left[0][0]))
            prob_right = 1 / (1 + np.exp(-output_right[0][0]))

            frame_closed = (prob_left < 0.5) and (prob_right < 0.5)

        # Száj kivágása
        mx1, my1 = mouth_left
        mx2, my2 = mouth_right
        mx, my = int((mx1 + mx2) / 2), int((my1 + my2) / 2)
        mw = int(np.linalg.norm(np.array(mouth_right) - np.array(mouth_left)))
        mouth_crop = frame[max(0, my - mw//2):min(frame.shape[0], my + mw//2),
                   max(0, mx - mw//2):min(frame.shape[1], mx + mw//2)]

        mouth_crop_rgb = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)

        if mouth_crop_rgb.size:
            input_mouth = transform_mouth(Image.fromarray(mouth_crop_rgb)).unsqueeze(0).numpy()
            output = mouth_session.run(None, {mouth_session.get_inputs()[0].name: input_mouth})[0]
            mouth_label = ["Normal", "Talk", "Yawn"][np.argmax(output)]


    # PERCLOS számítás
    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time

    if frame_closed:
        total_closed_time += delta_time
        eye_closed_continuous_time += delta_time
    else:
        eye_closed_continuous_time = 0.0

    elapsed_time = current_time - window_start_time
    perclos_value = total_closed_time / elapsed_time if elapsed_time > 0 else 0

    if elapsed_time >= window_duration:
        total_closed_time = 0.0
        window_start_time = current_time

    if perclos_value > closed_threshold or eye_closed_continuous_time >= continuous_closed_threshold:
        overall_label = "Alert"
        text_color = (0, 0, 255)

    # Képmegjelenítés
    frame_display = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.putText(frame_display, overall_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame_display, f"L: {prob_left:.2f} R: {prob_right:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_display, f"PERCLOS: {perclos_value*100:.1f}%", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_display, f"Mouth: {mouth_label}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # FPS
    fps = 1 / (time.time() - start_time_fps + 1e-8)
    cv2.putText(frame_display, f"FPS: {fps:.2f}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Fatigue Monitoring", frame_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
