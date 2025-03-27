import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transzformáció
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# CNN model
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

# Betöltés mentett súlyokkal
model = EyeCNN().to(device)
state_dict = torch.load("test3cnn.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# MTCNN arc detektálásra
mtcnn = MTCNN(keep_all=True, device=device)

# Kamera megnyitása
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    overall_label = "No face"
    text_color = (255, 255, 255)
    box_color = (255, 255, 255)

    prob_left = 0.0
    prob_right = 0.0
    box_left = (0, 0, 0, 0)
    box_right = (0, 0, 0, 0)

    if landmarks is not None:
        pts = landmarks[0]
        left_eye, right_eye = pts[0], pts[1]

        eye_dist = np.linalg.norm(left_eye - right_eye)
        roi_size = int(eye_dist * 0.7) # arányos ROI
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
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        
        # Kivágás
        roi_left, box_left = get_eye_roi(left_eye, frame, roi_size)
        roi_right, box_right = get_eye_roi(right_eye, frame, roi_size)

        if roi_left.size != 0 and roi_right.size != 0:

            # Szürkeárnyalatra konvertálás
            roi_left_gray = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
            roi_right_gray = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)

            # PIL formátumra alakítás
            roi_left_pil = Image.fromarray(roi_left_gray)
            roi_right_pil = Image.fromarray(roi_right_gray)

            # Transzformáció alkalmazása
            input_left = transform_test(roi_left_pil).unsqueeze(0).to(device)
            input_right = transform_test(roi_right_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output_left = model(input_left)
                output_right = model(input_right)
                prob_left = torch.sigmoid(output_left).item()
                prob_right = torch.sigmoid(output_right).item()
                pred_left = 1 if prob_left > 0.5 else 0
                pred_right = 1 if prob_right > 0.5 else 0

            # Eredmény 
            if prob_left > 0.7 and prob_right > 0.7:
                overall_label = "Open"
                text_color = (87, 140, 79)
                box_color = (180, 153, 72)

            elif prob_left < 0.3 and prob_right < 0.3:
                overall_label = "Closed"
                text_color = (26, 26, 163)
                box_color = (180, 153, 72)

            else:
                overall_label = "Uncertain"
                text_color = (85, 163, 227)
                box_color = (180, 153, 72)

            # Keretek
            cv2.rectangle(frame, (box_left[0], box_left[1]), (box_left[2], box_left[3]), box_color, 2)
            cv2.rectangle(frame, (box_right[0], box_right[1]), (box_right[2], box_right[3]), box_color, 2)

            # Valószínűségek
            cv2.putText(frame, f"L:{prob_left:.2f} R:{prob_right:.2f}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.putText(frame, overall_label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Megjelenítés
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    cv2.putText(frame_display, overall_label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    
    if landmarks is not None:
        cv2.putText(frame_display, f"L:{prob_left:.2f} R:{prob_right:.2f}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.rectangle(frame_display, (box_left[0], box_left[1]), (box_left[2], box_left[3]), box_color, 2)
        cv2.rectangle(frame_display, (box_right[0], box_right[1]), (box_right[2], box_right[3]), box_color, 2)

    cv2.imshow("Video Test", frame_display)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
