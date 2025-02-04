import imageio
import os
import shutil
import numpy as np
from scipy.stats import entropy

def calculate_mask_entropy(mask):
    """
    Calculate the entropy of a mask.
    """
    mask_np = mask.cpu().numpy().flatten()
    _, counts = np.unique(mask_np, return_counts=True)
    probabilities = counts / counts.sum()
    mask_entropy = entropy(probabilities, base=2)
    return mask_entropy

def filter_low_entropy(prompts):
    """
    Given a dictionary of prompts, filters out the ones with low entropy.
    """
    sorted_items = sorted(prompts.items(), key=lambda x: x[1])
    half_size = len(sorted_items) // 2
    filtered_prompts = {k:v for k, v in sorted_items[:half_size]}
    return filtered_prompts


def get_fps(filepath):
    """
    Returns the frames per second (fps) of a video file.
    """
    metadata = imageio.v3.immeta(filepath, exclude_applied=False)
    return metadata['fps']

def split_images(filepath, frames_dir):
    """
    Splits a video into frames and saves them as images in a specified directory.
    """
    reader = imageio.get_reader(filepath)

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)
    else:
        os.makedirs(frames_dir)
    
    for idx, frame in enumerate(reader):
        imageio.imwrite(frames_dir + f"/{idx:05d}.jpg", frame)