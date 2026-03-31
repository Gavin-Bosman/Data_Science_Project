import os
import re
import math
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import flat_transformer as ft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def expand_clips_spread(x, target_clips=7):
    """
    x: [C, 12, 512] where C=2
    returns: [7, 12, 512]
    """
    C = x.shape[0]

    if C == target_clips:
        return x

    # normalized positions across timeline
    indices = torch.linspace(0, C - 1, steps=target_clips)
    mapping = indices.round().long()

    return x[mapping]

class VideoFeatureDataset2(Dataset):
    def __init__(self, groups, video_ids, expected_clips=7, expected_frames=12, input_dim=512):
        self.groups = groups
        self.video_ids = video_ids
        self.expected_clips = expected_clips
        self.expected_frames = expected_frames
        self.input_dim = input_dim

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        clip_files = self.groups[vid]

        clip_tensors = []
        labels = []

        for f in clip_files:
            data = torch.load(f, map_location="cpu")
            feats = data["features"]   # expected [12, 512]
            label = data["label"]

            if feats.shape != (self.expected_frames, self.input_dim):
                raise ValueError(
                    f"Clip {os.path.basename(f)} has shape {tuple(feats.shape)}, "
                    f"expected {(self.expected_frames, self.input_dim)}"
                )

            clip_tensors.append(feats)
            labels.append(label)

        # all clips from the same video should have the same label
        if len(set(labels)) != 1:
            raise ValueError(f"Video {vid} has inconsistent clip labels: {labels}")

        x = torch.stack(clip_tensors, dim=0)   # [2, 12, 512]
        # Expand clips to match 7 
        x = expand_clips_spread(x, target_clips=self.expected_clips)
        y = labels[0]

        return x.float(), torch.tensor(y, dtype=torch.long)

feature_dir = Path(__file__).resolve().parent.parent / "data" / "features_enterface"
feature_dir = str(feature_dir)
vid_groups = ft.group_feature_files_by_video(feature_dir, expected_clips=2)
vid_ids = list(vid_groups.keys())
enterface_loader = DataLoader(
    VideoFeatureDataset2(vid_groups, vid_ids), 
    8,
    shuffle=False
)

model = ft.FlatTemporalTransformer(
    input_dim=512,
    d_model=256,
    num_classes=8,
    num_clips=7,
    frames_per_clip=12,
    nhead=4,
    num_layers=2,
    ff_dim=512,
    dropout=0.2
).to(device)

src_path = Path(__file__).resolve().parent
# Change last portion of path to current model config
weights_path = src_path / "transformer_weights" / "best_transformer_weights_1.pth"

model.load_state_dict(torch.load(weights_path))
model.eval()

RAVDESS_SUBSET = torch.tensor([2, 3, 4, 5, 6, 7])

# Load data and eval now with model

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    subset = RAVDESS_SUBSET.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            outputs = outputs[:, subset]
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # store for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc, all_preds, all_labels

test_loss, test_acc, all_preds, all_labels = evaluate(
    model,
    enterface_loader,
    criterion
)

print("\n===== eNTERFACE Evaluation =====")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.2f}%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)