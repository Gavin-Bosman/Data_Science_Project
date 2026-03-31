from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import mediapipe as mp
import cv2 as cv
from typing import Any
from pathlib import Path

def get_landmarker_task_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    task_path = base_dir / "mediapipe" / "face_landmarker.task"
    return str(task_path)
    
def get_face_landmarker(running_mode:str = "image", num_faces:int = 1, min_face_detection_confidence:float = 0.4, 
                        min_face_presence_confidence:float = 0.7, min_tracking_confidence:float = 0.7, 
                        output_face_blendshapes:bool = False, output_transform_matrixes:bool = False):
    task_path = get_landmarker_task_path()

    match running_mode.lower():
        case "image":
            running_mode = VisionTaskRunningMode.IMAGE
        case "video":
            running_mode = VisionTaskRunningMode.VIDEO
        case _:
            raise ValueError("Unrecognized value passed to parameter running_mode. Expects one of 'image' or 'video'.")

    baseOptions = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options = baseOptions,
        running_mode = running_mode,
        num_faces = num_faces,
        min_face_detection_confidence = min_face_detection_confidence,
        min_face_presence_confidence = min_face_presence_confidence,
        min_tracking_confidence = min_tracking_confidence,
        output_face_blendshapes = output_face_blendshapes,
        output_facial_transformation_matrixes = output_transform_matrixes
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    return detector

def get_pixel_coordinates(frame_rgb:cv.typing.MatLike, face_landmarker:Any) -> tuple[list[tuple[int,int]], list[Any] | None]:
    
    # Save the orignal dimensions for determining padding
    original_h, original_w = frame_rgb.shape[:2]

    # Pad to square dimensions before face landmarking
    if original_h > original_w:
        pad = (original_h - original_w) // 2
        padded_frame = cv.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=(0,0,0))

        vert_pad, horiz_pad = 0, pad
    elif original_w > original_h:
        pad = (original_w - original_h) // 2
        padded_frame = cv.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))

        vert_pad, horiz_pad = pad, 0
    else:
        padded_frame = frame_rgb
        vert_pad, horiz_pad = 0, 0
    
    # Get the new image dimensions after padding
    padded_h, padded_w = padded_frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_frame)

    lm_results = face_landmarker.detect(mp_image)
    
    if lm_results.face_landmarks:
        pixel_coords = []
        for lm in lm_results.face_landmarks[0]:
            x_pad = int(lm.x * padded_w)
            y_pad = int(lm.y * padded_h)

            x_pix = x_pad - horiz_pad
            y_pix = y_pad - vert_pad

            pixel_coords.append((x_pix, y_pix))
    else:
        return None
    
    return pixel_coords