import cv2 as cv
import numpy as np
import os
import pandas as pd
from pathlib import Path
from src.face_landmarker_utils import get_face_landmarker, get_pixel_coordinates

def get_directory_walk(input_directory:Path) -> pd.Series:
    full_file_paths = []

    for path, dirs, files in os.walk(str(input_directory), topdown=True):
        for file in files:
            full_path = os.path.join(path, file)
            
            full_file_paths.append(full_path)

    series1 = pd.Series(full_file_paths)

    return series1

# Extract all ravdess video paths
# First must unzip folder
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "Video_Speech_Actors_01-24"
paths = get_directory_walk(data_path)
output_folder_path = base_dir / "data" / "preprocessed_data"
face_landmarker = get_face_landmarker(
    running_mode="image",
    num_faces=1, 
    output_face_blendshapes=False
)

# Loop over each ravdess video
# Min duration 2.9362666
# Min frame count 88 - break each video into 7 clips of 12 frames (total 84)

for path in paths:
    _, file = os.path.split(path)
    filename, ext = os.path.splitext(file)
    name_tokens = str(filename).split("-")
    output_path = output_folder_path / f"Actor_{name_tokens[6]}"
    Path(str(output_path)).mkdir(exist_ok=True, parents=True)

    capture = cv.VideoCapture(path)
    writer = None
    clip_no = 1
    counter = 0

    while(capture.get(cv.CAP_PROP_POS_FRAMES) < 84):
        if counter % 12 == 0:
            if writer is not None:
                writer.release()
            writer = cv.VideoWriter(
                str(output_path / f"{filename}_clip-{clip_no}{ext}"),
                cv.VideoWriter.fourcc(*'mp4v'),
                29.97,
                (128, 128),
                isColor=True
            )
            clip_no += 1

        success, frame = capture.read()
        if not success:
            break

        lm_coords, _ = get_pixel_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_landmarker)
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