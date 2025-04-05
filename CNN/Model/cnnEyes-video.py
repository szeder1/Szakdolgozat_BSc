import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Teszt transzformáció
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# CNN modell
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Modell betöltése
model = EyeCNN().to(device)
state_dict = torch.load("test6cnn.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Arc- és szemdetektor
mtcnn = MTCNN(keep_all=True, device=device)

# Paraméterek
window_duration = 30 
closed_threshold = 0.4 
total_closed_time = 0.0
window_start_time = time.time()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    overall_label = "Monitoring"
    text_color = (255, 255, 255)

    prob_left = 0.0
    prob_right = 0.0

    current_time = time.time()
    frame_closed = False

    if landmarks is not None:
        pts = landmarks[0]
        left_eye, right_eye = pts[0], pts[1]

        eye_dist = np.linalg.norm(left_eye - right_eye)
        roi_size = int(eye_dist * 0.7)
        roi_size = max(20, min(roi_size, 100))

        def get_eye_roi(center, frame, roi_size):
            x1 = int(center[0] - roi_size // 2)
            y1 = int(center[1] - roi_size // 2)
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            return frame[y1:y2, x1:x2]

        roi_left = get_eye_roi(left_eye, frame, roi_size)
        roi_right = get_eye_roi(right_eye, frame, roi_size)

        if roi_left.size != 0 and roi_right.size != 0:
            roi_left_gray = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
            roi_right_gray = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)

            roi_left_pil = Image.fromarray(roi_left_gray)
            roi_right_pil = Image.fromarray(roi_right_gray)

            input_left = transform_test(roi_left_pil).unsqueeze(0).to(device)
            input_right = transform_test(roi_right_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output_left = model(input_left)
                output_right = model(input_right)
                prob_left = torch.sigmoid(output_left).item()
                prob_right = torch.sigmoid(output_right).item()

            closed = (prob_left < 0.5) and (prob_right < 0.5)
            frame_closed = closed

    # Időalapú PERCLOS számítás
    elapsed_time = current_time - window_start_time

    if frame_closed:
        total_closed_time += 1/30  # feltételezve kb. 30 FPS kamera

    if elapsed_time >= window_duration:
        perclos_value = total_closed_time / elapsed_time

        if perclos_value > closed_threshold:
            overall_label = "Alert"
            text_color = (0, 0, 255)
        
        # Ablak újraindítása
        total_closed_time = 0.0
        window_start_time = current_time

    else:
        # Folyamatosan is megjelenítjük a PERCLOS-t
        perclos_value = total_closed_time / elapsed_time if elapsed_time > 0 else 0

    # Képernyőre rajzolás
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Állapot kiírás
    cv2.putText(frame_display, overall_label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Bal és jobb szem valószínűségek
    cv2.putText(frame_display, f"L: {prob_left:.2f}   R: {prob_right:.2f}", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # PERCLOS érték
    cv2.putText(frame_display, f"PERCLOS: {perclos_value*100:.1f} %", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Fatigue Monitoring", frame_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
