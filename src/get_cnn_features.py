import zipfile
import os
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Unzip and extract input data ------------------------
root_path = Path(__file__).resolve().parent.parent
data_path = root_path / "data" / "preprocessed_data_enterface"
base_save_path = root_path / "data" / "features_enterface"

# zip_path = root_path / "data" / "preprocessed_data.zip"
# extract_path = root_path / "data" / "preprocessed_data"
# Path(extract_path).mkdir(exist_ok=True)

# if os.path.exists(zip_path):
#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_path)
#         print(f"Successfully extracted {zip_path} to {extract_path}")
#     except zipfile.BadZipFile:
#         print(f"Error: The zip file {zip_path} is corrupted.")
#     except Exception as e:
#         print(f"An error occurred during extraction: {e}")
# else:
#     print(f"Error: Zip file not found at {zip_path}")

def extract_12_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame).permute(2,0,1).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        frame = (frame - mean) / std

        frames.append(frame)

    cap.release()
    return torch.stack(frames)

def get_label_from_filename_ravdess(path):
    filename = os.path.basename(path)
    parts = filename.split('-')
    return int(parts[2]) - 1

# Subset of ravdess class labels 2-7
def get_label_from_filename_enterface(path):
    emo_map = {"ha": 0, "sa": 1, "an": 2, "fe": 3, "di": 4, "su": 5}
    filename, _ = os.path.splitext(os.path.basename(path))
    parts = str(filename).split("_")
    return emo_map.get(parts[1])
    
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

for actor in os.listdir(data_path):
    actor_path = data_path / actor

    if not os.path.isdir(actor_path):
        continue
    if actor not in {'Actor_6', 'Actor_7', 'Actor_8', 'Actor_9'}:
        continue

    for video_file in os.listdir(str(actor_path)):
        if video_file.endswith(".mp4"):

            video_path = os.path.join(str(actor_path), video_file)

            frames = extract_12_frames(video_path).to(device)

            with torch.no_grad():
                features = resnet(frames)

            label = get_label_from_filename_enterface(video_path)

            save_path = base_save_path / video_file.replace(".mp4", ".pt")

            torch.save({
                "features": features.cpu(),
                "label": label
            }, str(save_path))

            print(f"Saved {save_path}, shape: {features.shape}")