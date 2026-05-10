import cv2 as cv
import numpy as np

VIDEO_PATH = "data/input_video.mp4"

ALPHA = 0.5      # learning rate per il background update
THRESH = 20      # soglia sul frame differencing
KERNEL = np.ones((3, 3), np.uint8)
WARMUP_FRAMES = 30

cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERRORE: non riesco ad aprire il video:", VIDEO_PATH)
    exit(1)

background = None
frame_idx = 0
prev_centroids = None
MAX_MOVE = 200  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # inizializza background con il primo frame
    if background is None:
        background = gray.astype("float")
        frame_idx += 1
        continue

    # frame differencing: |frame - background|
    bg_uint8 = cv.convertScaleAbs(background)           # float -> uint8
    diff = cv.absdiff(gray, bg_uint8)
    diff_blur = cv.GaussianBlur(diff, (5, 5), 0)
    # soglia per ottenere foreground grezzo
    _, fgmask = cv.threshold(diff, THRESH, 255, cv.THRESH_BINARY)

    kernel_big = np.ones((5, 5), np.uint8)

    # prima pulisci il rumore fine
    mask_open = cv.morphologyEx(fgmask, cv.MORPH_OPEN, KERNEL)

    # poi chiudi i buchi interni nella sagoma
    mask_refined = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_big, iterations=2)

    mask_refined = cv.dilate(mask_refined, kernel_big, iterations=1)

    if frame_idx >= WARMUP_FRAMES:
        # raffinamento della maschera
        mask_open = cv.morphologyEx(fgmask, cv.MORPH_OPEN, KERNEL)
        mask_refined = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, KERNEL)

        # connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_refined)

        candidates = []
        for i in range(1, num_labels):  # 0 = background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            candidates.append((area, x, y, w, h, cx, cy))

        # ordina per area decrescente
        candidates.sort(key=lambda c: c[0], reverse=True)

        # seleziona al massimo 2 componenti, ma solo se distanti tra loro
        MIN_DIST = 300  # distanza minima tra centroidi in pixel (da tarare)
        components = []

        for cand in candidates:
            area, x, y, w, h, cx, cy = cand
            if not components:
                components.append(cand)
            else:
                # distanza dal primo componente già scelto
                _, _, _, _, _, cx0, cy0 = components[0]
                dist = np.hypot(cx - cx0, cy - cy0)
                if dist >= MIN_DIST:
                    components.append(cand)
            if len(components) == 2:
                break

        # ora 'components' contiene al massimo 2 blob più grandi e ben separati
        for player_id, (area, x, y, w, h, cx, cy) in enumerate(components, start=1):
            color = (0, 255, 0) if player_id == 1 else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)

    # aggiorna il background con running average
    cv.accumulateWeighted(gray, background, ALPHA)

    vis = cv.resize(frame, (1920, 1080))
    cv.imshow("Player detection", vis)
    cv.imshow("FG mask", cv.resize(fgmask, (1920, 1080)))

    key = cv.waitKey(20) & 0xFF
    if key in (ord('q'), ord('Q')):
        break

    frame_idx += 1

cap.release()
cv.destroyAllWindows()