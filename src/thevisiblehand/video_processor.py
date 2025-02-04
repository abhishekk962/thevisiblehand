import imageio
import glob
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from thevisiblehand.utils import calculate_mask_entropy, filter_low_entropy, get_fps, split_images
from thevisiblehand.hand_detection import create_detector, extract_hand_coordinates, show_mask, preview_results
from thevisiblehand.hand_masking import build_predictor, propagate_in_video


class VideoProcessor():
    """
    A class to process a video file, detect hands using Mediapipe, and mask them using the Segment Anything Model 2 (SAM2).
    """
    def __init__(self, in_filepath, out_filepath, num_hands=2, method='single'):
        self.in_filepath = in_filepath
        self.out_filepath = out_filepath
        self.num_hands = num_hands
        self.method = method
        self.predictor = build_predictor()
        self.detector = create_detector(num_hands)
        self.video_dir = "./frames"

        split_images(self.in_filepath, self.video_dir)

        self.frame_names = sorted(glob.glob(os.path.join(self.video_dir, "*.jpg")))

        self.inference_state = self.predictor.init_state(video_path=self.video_dir)

    def process_video(self):
        """
        Process the video by detecting hands, masking them and saving the output in the specified directory.
        """
        if self.method == 'single':
            add_unique_prompts_to_each_hand(self.detector, self.inference_state, self.frame_names, self.predictor)
        elif self.method == 'multi':
            add_prompts_to_multiple_frames(self.detector, self.inference_state, self.frame_names, self.predictor)
    
        video_segments = propagate_in_video(self.predictor, self.inference_state)

        fps = get_fps(self.in_filepath)

        save_output_video(video_segments, out_filepath=self.out_filepath, frame_names=self.frame_names, fps=fps)

        self.predictor.reset_state(self.inference_state)
        
    def preview_results(self):
        """
        Preview the results of the hand detection and masking and save the preview in the specified directory.
        """
        if self.method == 'single':
            results = add_unique_prompts_to_each_hand(self.detector, self.inference_state, self.frame_names, self.predictor)
        elif self.method == 'multi':
            results = add_prompts_to_multiple_frames(self.detector, self.inference_state, self.frame_names, self.predictor)

        preview_results(results, self.in_filepath, self.frame_names, self.method)

        self.predictor.reset_state(self.inference_state)
    

def save_output_video(video_segments, out_filepath, frame_names, fps):
    """
    Save the output video with masked hands.
    """
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    with imageio.get_writer(out_filepath, fps=fps) as writer:
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names)):
            plt.figure(figsize=(6, 4))
            plt.axis("off")
            plt.imshow(Image.open(frame_names[out_frame_idx]))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            buf = BytesIO()
            plt.savefig(buf, dpi=300, format='jpg', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img_array = np.array(Image.open(buf))
            writer.append_data(img_array)
            plt.close()
    print(f"Output video saved at {out_filepath}")

def add_unique_prompts_to_each_hand(detector, inference_state, frame_names, predictor):
    """
    Add prompts with unique object ID to each hand detected in the first frame.
    """
    ann_frame_idx = 0
    ann_obj_id = 0
    prompts = {}
    predictor.reset_state(inference_state)
    image = mp.Image.create_from_file(frame_names[ann_frame_idx])
    detection_result = detector.detect(image)
    pts = extract_hand_coordinates(image.numpy_view(), detection_result)
    # Add a unique object ID for each hand
    for pt in pts:
        ann_obj_id += 1
        points = np.array([pt], dtype=np.float32)
        labels = np.array([1], np.int32)
        prompts[ann_obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    results = [{
        "ann_frame_idx": ann_frame_idx,
        "prompts": prompts,
        "out_obj_ids": out_obj_ids,
        "out_mask_logits": out_mask_logits,
    }]

    return results

def add_prompts_to_multiple_frames(detector, inference_state, frame_names, predictor):
    """
    Add prompts to multiple frames and select frames which have lower mask entropy for actual prompting.
    """
    # Add prompts to 5 frames
    ann_obj_id = 1
    frames = {}
    predictor.reset_state(inference_state)
    for ann_frame_idx in range(0,len(frame_names),int(len(frame_names)/5)):
        prompts = {}
        image = mp.Image.create_from_file(frame_names[ann_frame_idx])
        detection_result = detector.detect(image)
        pts = extract_hand_coordinates(image.numpy_view(), detection_result)
        points = np.array(pts, dtype=np.float32)
        labels = np.array([1]*len(points), np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        frames[ann_frame_idx] = calculate_mask_entropy(out_mask_logits)
        prompts[ann_obj_id] = points, labels
    
    # Filter out frames with low entropy
    frames = filter_low_entropy(frames)
    predictor.reset_state(inference_state)
    results = []

    # Use the filtered frames for prompting
    for ann_frame_idx in frames:
        prompts = {}
        image = mp.Image.create_from_file(frame_names[ann_frame_idx])
        detection_result = detector.detect(image)
        pts = extract_hand_coordinates(image.numpy_view(), detection_result)
        points = np.array(pts, dtype=np.float32)
        labels = np.array([1]*len(points), np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        prompts[ann_obj_id] = points, labels

        results.append({
            "ann_frame_idx": ann_frame_idx,
            "prompts": prompts,
            "out_obj_ids": out_obj_ids,
            "out_mask_logits": out_mask_logits,
        })

    return results
