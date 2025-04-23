import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnxruntime as ort
import time
import mediapipe as mp
from collections import deque

# MediaPipe Face Mesh inicializálás
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ONNX modellek betöltése
eye_session = ort.InferenceSession("eye_model.onnx", providers=['CPUExecutionProvider'])
mouth_session = ort.InferenceSession("mouth_model.onnx", providers=['CPUExecutionProvider'])

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

# PERCLOS paraméterek
window_duration = 30
closed_threshold = 0.4
continuous_closed_threshold = 2.0
total_closed_time = 0.0
window_start_time = time.time()
eye_closed_continuous_time = 0.0
prev_time = time.time()

# FPS ellenőrzés
fps_window = deque(maxlen=300)

# Kamera
cap = cv2.VideoCapture(0)


while True:
    loop_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    overall_label = "Monitoring"
    text_color = (255, 255, 255)
    prob_left = prob_right = 0.0
    mouth_label = "Unknown"
    frame_closed = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        def coord(idx): return int(lm[idx].x * w), int(lm[idx].y * h)
        left_eye, right_eye = coord(33), coord(263)
        mouth_left, mouth_right = coord(78), coord(308)

        # Szem ROI
        dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        roi_size = max(20, min(int(dist * 0.7), 100))

        def crop(c):
            x, y = c
            x1, y1 = max(0, x - roi_size // 2), max(0, y - roi_size // 2)
            return frame[y1:y1+roi_size, x1:x1+roi_size]

        roi_l, roi_r = crop(left_eye), crop(right_eye)

        if roi_l.size and roi_r.size:
            in_l = transform_eye(Image.fromarray(cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY))).unsqueeze(0).numpy()
            in_r = transform_eye(Image.fromarray(cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY))).unsqueeze(0).numpy()
            out_l = eye_session.run(None, {eye_session.get_inputs()[0].name: in_l})[0]
            out_r = eye_session.run(None, {eye_session.get_inputs()[0].name: in_r})[0]
            prob_left = 1 / (1 + np.exp(-out_l[0][0]))
            prob_right = 1 / (1 + np.exp(-out_r[0][0]))
            frame_closed = (prob_left < 0.5 and prob_right < 0.5)

        # Száj ROI
        mx = (mouth_left[0] + mouth_right[0]) // 2
        my = (mouth_left[1] + mouth_right[1]) // 2
        mw = int(np.linalg.norm(np.array(mouth_right) - np.array(mouth_left)))
        mroi = frame[max(0, my - mw // 2):min(h, my + mw // 2), max(0, mx - mw // 2):min(w, mx + mw // 2)]

        if mroi.size:
            in_m = transform_mouth(Image.fromarray(cv2.cvtColor(mroi, cv2.COLOR_BGR2RGB))).unsqueeze(0).numpy()
            out_m = mouth_session.run(None, {mouth_session.get_inputs()[0].name: in_m})[0]
            mouth_label = ["Normal", "Talk", "Yawn"][np.argmax(out_m)]

    # PERCLOS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    total_closed_time += dt if frame_closed else 0.0
    eye_closed_continuous_time = eye_closed_continuous_time + dt if frame_closed else 0.0
    elapsed_w = now - window_start_time
    if elapsed_w >= window_duration:
        total_closed_time = 0.0
        window_start_time = now
    perclos = total_closed_time / elapsed_w if elapsed_w > 0 else 0.0
    if perclos > closed_threshold or eye_closed_continuous_time >= continuous_closed_threshold:
        overall_label, text_color = "Alert", (0, 0, 255)

    # FPS számítás és naplózás
    loop_end = time.perf_counter()
    fps = 1.0 / max(loop_end - loop_start, 1e-6)
    fps_window.append(fps)

    # FPS ellenőrzés
    below_30_count = len([f for f in fps_window if f < 30])
    warning_text = ""
    if len(fps_window) >= 30 and below_30_count / len(fps_window) > 0.3:
        warning_text = "Alacsony FPS!"

    avg_fps_text = ""
    if len(fps_window) >= 30:
        avg_fps = sum(fps_window) / len(fps_window)
        avg_fps_text = f"Átlag FPS: {avg_fps:.2f}"
        
    # Megjelenítés
    disp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    cv2.putText(disp, overall_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(disp, f"L:{prob_left:.2f} R:{prob_right:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(disp, f"PERCLOS:{perclos*100:.1f}%", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(disp, f"Mouth:{mouth_label}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(disp, f"FPS: {fps:.1f}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    
    if warning_text:
        cv2.putText(disp, warning_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Megjelenítés
    cv2.imshow('Fatigue Monitoring', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
