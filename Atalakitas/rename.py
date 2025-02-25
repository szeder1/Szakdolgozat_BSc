import os

frame_directory = 'E:\\Egyetem\\Szakdolgozat\\Final\\Datasets\\yawdd\\Frame\\v3'

for folder in os.listdir(frame_directory):
    folder_path = os.path.join(frame_directory, folder)
    
    if os.path.isdir(folder_path):
        images = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        temp_names = {}

        for index, image in enumerate(images, start=1):
            old_path = os.path.join(folder_path, image)
            temp_path = os.path.join(folder_path, f"temp_{index}.png")

            os.rename(old_path, temp_path)
            temp_names[temp_path] = os.path.join(folder_path, f"frame_{index}.png")

        for temp_path, final_path in temp_names.items():
            os.rename(temp_path, final_path)

        print(f"{folder}: Képek sikeresen átnevezve!")
