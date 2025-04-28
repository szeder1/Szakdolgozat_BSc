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
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ONNX modellek betöltése
eye = ort.InferenceSession("eye_model.onnx", providers=['CPUExecutionProvider'])
mouth = ort.InferenceSession("mouth_model.onnx", providers=['CPUExecutionProvider'])

# Transzformációk
transform_eye = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
transform_mouth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# PERCLOS paraméterek
perclos_idotartam = 30
szemzaras_kuszob = 0.4
folyamatos_zaras_idokuszob = 2.0
ossz_szemzaras_ido = 0.0
perclos_ablak_kezdete = time.time()
szem_folyamatos_zaras_ido = 0.0
utolso_idopont = time.time()

# Átlag feldolgozási FPS mérés
fps_lista = deque(maxlen=300)

# Kamera beállítása
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
kamera.set(cv2.CAP_PROP_FPS, 30)

# Utolsó frame tárolása
utolso_kep_hash = None

# Feldolgozási FPS korlát
ciklus_ido = 1.0 / 30.0 

# Tárolt eredmények
bal_szem_valoszinuseg = jobb_szem_valoszinuseg = 0.0
aktualis_szaj_allapot = "Unknown"
aktualis_szem_zarva = False

while True:
    loop_start = time.perf_counter()

    sikeres, frame = kamera.read()
    if not sikeres:
        break

    # Csak új frame feldolgozása
    aktualis_kep_hash = hash(frame.tobytes())
    if aktualis_kep_hash == utolso_kep_hash:
        time.sleep(0.001)
        continue
    utolso_kep_hash = aktualis_kep_hash

    # Face Mesh
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    figyelmeztetes = "Monitoring"
    text_color = (255, 255, 255)

    if results.multi_face_landmarks:
        landmarkok = results.multi_face_landmarks[0].landmark
        magassag, szelesseg, _ = frame.shape
        def coord(idx): return int(landmarkok[idx].x * szelesseg), int(landmarkok[idx].y * magassag)
        left_eye, right_eye = coord(33), coord(263)
        mouth_left, mouth_right = coord(78), coord(308)

        # Szem ROI
        tavolsag = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        roi_meret = max(20, min(int(tavolsag * 0.7), 100))

        left_roi_x, left_roi_y = max(0, left_eye[0] - roi_meret//2), max(0, left_eye[1] - roi_meret//2)
        bal_szem_roi = frame[left_roi_y:left_roi_y+roi_meret, left_roi_x:left_roi_x+roi_meret]

        right_roi_x, right_roi_y = max(0, right_eye[0] - roi_meret//2), max(0, right_eye[1] - roi_meret//2)
        jobb_szem_roi = frame[right_roi_y:right_roi_y+roi_meret, right_roi_x:right_roi_x+roi_meret]

        if bal_szem_roi.size and jobb_szem_roi.size:
            input_l = transform_eye(Image.fromarray(cv2.cvtColor(bal_szem_roi, cv2.COLOR_BGR2GRAY)))\
                   .unsqueeze(0).numpy()
            input_r = transform_eye(Image.fromarray(cv2.cvtColor(jobb_szem_roi, cv2.COLOR_BGR2GRAY)))\
                   .unsqueeze(0).numpy()
            out_l = eye.run(None, {eye.get_inputs()[0].name: input_l})[0]
            out_r = eye.run(None, {eye.get_inputs()[0].name: input_r})[0]
            bal_szem_valoszinuseg = 1/(1+np.exp(-out_l[0][0]))
            jobb_szem_valoszinuseg = 1/(1+np.exp(-out_r[0][0]))
            aktualis_szem_zarva = (bal_szem_valoszinuseg<0.5 and jobb_szem_valoszinuseg<0.5)

        # Száj ROI
        szaj_x, szaj_y = (mouth_left[0]+mouth_right[0])//2, (mouth_left[1]+mouth_right[1])//2
        szaj_w = int(np.linalg.norm(np.array(mouth_right)-np.array(mouth_left)))
        szaj_roi_x, szaj_roi_y = max(0, szaj_x-szaj_w//2), max(0, szaj_y-szaj_w//2)
        szaj_roi = frame[szaj_roi_y:szaj_roi_y+szaj_w, szaj_roi_x:szaj_roi_x+szaj_w]
        if szaj_roi.size:
            input_m = transform_mouth(Image.fromarray(cv2.cvtColor(szaj_roi, cv2.COLOR_BGR2RGB)))\
                   .unsqueeze(0).numpy()
            out_m = mouth.run(None, {mouth.get_inputs()[0].name: input_m})[0]
            aktualis_szaj_allapot = ["Normal","Talk","Yawn"][np.argmax(out_m)]

    # PERCLOS
    now = time.time()
    idokulonbseg = now - utolso_idopont
    utolso_idopont = now
    ossz_szemzaras_ido += idokulonbseg if aktualis_szem_zarva else 0.0
    szem_folyamatos_zaras_ido += idokulonbseg if aktualis_szem_zarva else 0.0
    eltelt_ido = now - perclos_ablak_kezdete

    if eltelt_ido >= perclos_idotartam:
        ossz_szemzaras_ido = 0.0
        perclos_ablak_kezdete  = now

    perclos_arany = ossz_szemzaras_ido /eltelt_ido  if eltelt_ido >0 else 0

    if perclos_arany > szemzaras_kuszob  or szem_folyamatos_zaras_ido >=folyamatos_zaras_idokuszob:
        figyelmeztetes, text_color = "Alert", (0,0,255)

    # Megjelenítés
    kijelzo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kijelzo = cv2.cvtColor(kijelzo, cv2.COLOR_GRAY2BGR)

    cv2.putText(kijelzo, figyelmeztetes, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, text_color,2)
    cv2.putText(kijelzo, f"L:{bal_szem_valoszinuseg:.2f} R:{jobb_szem_valoszinuseg:.2f}", (50,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    cv2.putText(kijelzo, f"PERCLOS:{perclos_arany*100:.1f}%", (50,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    cv2.putText(kijelzo, f"Mouth:{aktualis_szaj_allapot}", (50,170),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    # Cikluskorlátozás: max 30 FPS feldolgozás
    loop_end = time.perf_counter()
    ciklus_hossz = loop_end - loop_start
    if ciklus_hossz < ciklus_ido:
        time.sleep(ciklus_ido - ciklus_hossz)
        loop_end = time.perf_counter()
        ciklus_hossz = loop_end - loop_start

    # Átlag feldolgozási FPS számítása
    aktualis_fps = 1.0 / max(ciklus_hossz, 1e-6)
    fps_lista.append(aktualis_fps)
    if len(fps_lista) >= 10:
        atlag_fps = sum(fps_lista)/len(fps_lista)
        cv2.putText(kijelzo, f"Atlag FPS: {atlag_fps:.2f}",
                    (50,210), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    cv2.imshow('Fatigue Monitoring', kijelzo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
