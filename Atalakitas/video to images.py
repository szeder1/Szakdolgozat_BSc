import cv2
import os

video_directory = 'E:\\Egyetem\\Szakdolgozat\\Final\Datasets\\yawdd\\YawDD dataset\\Mirror\\Male_mirror Avi Videos'
output_base_folder = 'E:\\Egyetem\\Szakdolgozat\\Final\\Datasets\\yawdd\\Frame\\v3'
frame_rate = 3 
allowed_prefixes = ('2', '6', '7', '10', '11', '12', '17', '18', '19', '20', '21', '22', '24', '25', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40', '41', '42', '44', '45', '46', '47')

os.makedirs(output_base_folder, exist_ok=True)

for filename in os.listdir(video_directory):
    if filename.endswith('.avi'):
        if any(filename.startswith(f"{prefix}-") for prefix in allowed_prefixes):
            video_path = os.path.join(video_directory, filename)
            output_folder = os.path.join(output_base_folder, filename.split('.')[0])
            os.makedirs(output_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Nem sikerült megnyitni a videót: {video_path}")
                continue

            frame_count = 0
            saved_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_rate == 0:
                    frame_name = os.path.join(output_folder, f"frame_{frame_count}.png")
                    cv2.imwrite(frame_name, frame)
                    saved_frames += 1
                    print(f"Mentve: {frame_name}")

                frame_count += 1

            cap.release()
            print(f"Összesen {saved_frames} képkocka lett mentve a következő videóból: {filename}")


