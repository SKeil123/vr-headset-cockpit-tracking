import cv2
from pathlib import Path

#Pfade
video_dir = Path(r"C:\Users\shko0\cockpit_videos\raw")
frame_dir = Path(r"C:\Users\shko0\cockpit_videos\frames")
frame_dir.mkdir(parents=True, exist_ok=True)

frame_step = 10     # jedes 10.Frame

video_exts = [".mp4", ".MOV", ".mov"]

for video_path in sorted(video_dir.iterdir()):
    if video_path.suffix not in video_exts:
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Konnte {video_path.name} nicht öffnen")
        continue

    out_dir = frame_dir / video_path.stem
    out_dir.mkdir(exist_ok=True)

    i = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_step == 0:
            cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), frame)
            saved += 1

        i += 1
    
    cap.release()
    print(f"{video_path.name}: {saved} Frames extrahiert")
