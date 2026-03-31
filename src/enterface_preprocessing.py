import cv2 as cv
import numpy as np
import av
import os
import pandas as pd
from pathlib import Path
import face_landmarker_utils as mputil

# Extract all ravdess video paths
# First must unzip folder
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "testing" / "enterface_mp4"
output_folder_path = base_dir / "data" / "preprocessed_data_enterface"
face_landmarker = mputil.get_face_landmarker(
    running_mode="image",
    num_faces=1, 
    output_face_blendshapes=False
)

files = list(data_path.rglob("*.mp4"))
files = [p.resolve() for p in files]

# Loop over each eNTERFACE video
# Min frame count 28.0 - break each video into 2 clips of 12 frames (total 24)

for i, path in enumerate(files):
    _, file = os.path.split(path)
    filename, ext = os.path.splitext(file)
    name_tokens = str(filename).split("_")
    output_path = output_folder_path / f"Actor_{name_tokens[0][1:]}"
    Path(str(output_path)).mkdir(exist_ok=True, parents=True)

    capture = cv.VideoCapture(path)
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)

    writer = None
    clip_no = 1
    counter = 0

    while(capture.get(cv.CAP_PROP_POS_FRAMES) < 24):
        if counter % 12 == 0:
            if writer is not None:
                writer.release()
            writer = cv.VideoWriter(
                str(output_path / f"{filename}_clip-{clip_no}{ext}"),
                cv.VideoWriter.fourcc(*'mp4v'),
                fps,
                (128, 128),
                isColor=True
            )
            clip_no += 1

        success, frame = capture.read()
        if not success:
            break

        lm_coords = mputil.get_pixel_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_landmarker)
        if lm_coords is None:
            print(f"Failed to find face in {filename}")
            break
        pts = np.array(lm_coords, dtype=np.int32)

        x,y,w,h = cv.boundingRect(pts)
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)

        x = x - pad_x
        y = y - pad_y
        padded_width = w + 2*pad_x
        padded_height = h + 2*pad_y

        # Compute padded letterbox
        size = max(padded_width, padded_height)
        padded_letterbox = np.ones((size, size, 3), dtype=np.uint8) * 255

        # Resize dims to include padding
        cx = x + padded_width // 2
        cy = y + padded_height // 2
        x = cx - size // 2
        y = cy - size // 2
        w = h = size
        img_h, img_w = frame.shape[:2]

        # Coords in original image
        x1_src = max(0, x)
        y1_src = max(0, y)
        x2_src = min(x + size, img_w)
        y2_src = min(y + size, img_h)

        # Coords in padded letterbox
        x1_dst = x1_src - x
        y1_dst = y1_src - y
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        padded_letterbox[y1_dst:y2_dst, x1_dst:x2_dst] = frame[y1_src:y2_src, x1_src:x2_src]
        resized = cv.resize(padded_letterbox, (128, 128), interpolation=cv.INTER_AREA)

        counter += 1
        writer.write(resized)
    
    capture.release()
    writer.release()
    if i % 100 == 0:
        print(f"Processed file {i}")