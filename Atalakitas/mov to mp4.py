import cv2
import os
import time

video_mov = 'E:\\Egyetem\\Szakdolgozat\\Final\\Sajat\\Base\\1.mov'
video_mp4 = 'E:\\Egyetem\\Szakdolgozat\\Final\\Sajat\\Base\\1.mp4'

cap = cv2.VideoCapture(video_mov)
if not cap.isOpened():
    print("Nem sikerült megnyitni a MOV fájlt.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_mp4, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    out.write(frame)

cap.release()
out.release()

print("MOV → MP4 konvertálás kész.")

output_folder = 'E:\\Egyetem\\Szakdolgozat\\Final\\Sajat\\Base\\Frame\\Egy'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_mp4)
if not cap.isOpened():
    print("Nem sikerült megnyitni az MP4 fájlt.")
    exit()

frame_rate = 3
frame_count = 0
saved_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_rate == 0:
        saved_frames += 1
        frame_name = os.path.join(output_folder, f"frame_{saved_frames}.png")
        if cv2.imwrite(frame_name, frame):
            print(f"Mentve: {frame_name}")
        else:
            print(f"Hiba a mentésnél: {frame_name}")
        #time.sleep(0.5)

    frame_count += 1

cap.release()
print(f"Képkockák kivágása kész. Összesen {saved_frames} képkocka lett mentve.")
