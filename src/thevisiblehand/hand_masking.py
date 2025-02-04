import torch
import os
import urllib.request
from sam2.build_sam import build_sam2_video_predictor
from functools import cache

@cache
def build_predictor():
    """
    Build the SAM2 video predictor.
    """
    # Check for available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")

    # Download the model if not already present
    os.makedirs('../checkpoints/', exist_ok=True)

    if not os.path.exists('../checkpoints/sam2.1_hiera_large.pt'):
        url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
        urllib.request.urlretrieve(url, '../checkpoints/sam2.1_hiera_large.pt')

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Build the predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    return predictor


def propagate_in_video(predictor, inference_state):
    """
    Propagate the predictions in the video.
    """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments