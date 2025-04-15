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

# Transzformáció
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

# Szemállapot-klasszifikáló hálózat
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Szájállapot-klasszifikáló hálózat
class MouthCNN(nn.Module):
    def __init__(self):
        super(MouthCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Modellek betöltése
eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load("test14cnn.pth", map_location=device))
eye_model.eval()

mouth_model = MouthCNN().to(device)
mouth_model.load_state_dict(torch.load("test2m.pth", map_location=device))
mouth_model.eval()

# Arc detektálás
mtcnn = MTCNN(keep_all=True, device=device)

# Paraméterek
window_duration = 30
closed_threshold = 0.4
continuous_closed_threshold = 2.0

total_closed_time = 0.0
window_start_time = time.time()
eye_closed_continuous_time = 0.0
prev_time = time.time()
landmarks_last = None
frame_count = 0

# Kamera megnyitása
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_count += 1

    if frame_count % 5 == 0 or landmarks_last is None:
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        if landmarks is not None:
            landmarks_last = landmarks
    else:
        landmarks = landmarks_last

    overall_label = "Monitoring"
    text_color = (255, 255, 255)
    prob_left, prob_right = 0.0, 0.0
    mouth_label = "Unknown"
    frame_closed = False

    with torch.no_grad():
        if landmarks is not None and boxes is not None:
            pts = landmarks[0]
            left_eye, right_eye = pts[0], pts[1]

            # Szem környékének kivágása
            def get_eye_roi(center, frame, roi_size):
                x1 = int(center[0] - roi_size // 2)
                y1 = int(center[1] - roi_size // 2)
                x2, y2 = x1 + roi_size, y1 + roi_size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                return frame[y1:y2, x1:x2]
            
            # ROI mérete
            eye_dist = np.linalg.norm(left_eye - right_eye)
            roi_size = int(eye_dist * 0.7)
            roi_size = max(20, min(roi_size, 100))

            roi_left = get_eye_roi(left_eye, frame, roi_size)
            roi_right = get_eye_roi(right_eye, frame, roi_size)

            if roi_left.size != 0 and roi_right.size != 0:
                roi_left_gray = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
                roi_right_gray = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)

                input_left = transform_eye(Image.fromarray(roi_left_gray)).unsqueeze(0).to(device)
                input_right = transform_eye(Image.fromarray(roi_right_gray)).unsqueeze(0).to(device)

                output_left = eye_model(input_left)
                output_right = eye_model(input_right)

                prob_left = torch.sigmoid(output_left).item()
                prob_right = torch.sigmoid(output_right).item()

                frame_closed = (prob_left < 0.5) and (prob_right < 0.5)

            # Száj kivágása
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                width = x2 - x1
                height = y2 - y1

                mouth_x1 = x1 + int(width * 0.2)
                mouth_x2 = x2 - int(width * 0.2)
                mouth_y1 = y1 + int(height * 0.6)
                mouth_y2 = y2

                mouth_x1 = max(0, mouth_x1)
                mouth_y1 = max(0, mouth_y1)
                mouth_x2 = min(frame.shape[1], mouth_x2)
                mouth_y2 = min(frame.shape[0], mouth_y2)

                mouth_crop = frame_rgb[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            if mouth_crop.size != 0:
                    input_mouth = transform_mouth(Image.fromarray(mouth_crop)).unsqueeze(0).to(device)
                    output = mouth_model(input_mouth)
                    pred_class = output.argmax(dim=1).item()
                    mouth_label = ["Normal", "Talk", "Yawn"][pred_class]

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

    if (perclos_value > closed_threshold) or (eye_closed_continuous_time >= continuous_closed_threshold):
        overall_label = "Alert"
        text_color = (0, 0, 255)


    # Eredmény megjelenítés
    frame_display = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    cv2.putText(frame_display, overall_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame_display, f"L: {prob_left:.2f} R: {prob_right:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_display, f"PERCLOS: {perclos_value*100:.1f}%", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_display, f"Mouth: {mouth_label}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Fatigue Monitoring", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

