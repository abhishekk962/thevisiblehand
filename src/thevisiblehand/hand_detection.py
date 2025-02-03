import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functools import cache

@cache
def create_detector(num_hands):
    os.makedirs('../checkpoints/', exist_ok=True)

    if not os.path.exists('../checkpoints/hand_landmarker.task'):
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, '../checkpoints/hand_landmarker.task')

    base_options = python.BaseOptions(model_asset_path='../checkpoints/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=num_hands)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def get_hand_detection_result(imagepath, detector):
    image = mp.Image.create_from_file(imagepath)
    detection_result = detector.detect(image)
    return image, detection_result

def extract_hand_coordinates(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    height, width, _ = rgb_image.shape

    coordinates = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        coordinates.append([hand_landmarks[0].x*width,hand_landmarks[0].y*height])

    return coordinates


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def preview_results(results, in_filepath, frame_names, method):
    for r, res in enumerate(results):
        plt.figure(figsize=(9, 6))
        plt.axis("off")
        plt.imshow(Image.open(frame_names[res["ann_frame_idx"]]))
        for i, out_obj_id in enumerate(res["out_obj_ids"]):
            points = res["prompts"][out_obj_id]
            show_points(*points, plt.gca())
            show_mask((res["out_mask_logits"][i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.savefig(in_filepath+f'-{method}-preview-{r+1}.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()