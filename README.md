# TheVisibleHand

## Description
This repository showcases a method for segmenting hands in videos using Google's MediaPipe and Meta's SAM2 model. By leveraging MediaPipe's accurate hand position detection and SAM2's object segmentation across multiple frames, this project provides an effective way to isolate hands with minimal prompting. Additionally, it introduces enhancements that reduce artifacts in the masked video through an entropy-based filtering approach, which removes ill-formed masks and improves mask quality.

### Methods for Masking

#### **Method 1: Single-Frame Prompting**
This approach uses separate point prompts to identify each hand in a single frame. The SAM2 model then tracks and segments the hands throughout the video, ensuring that each hand is uniquely masked.

<div align="center"><img src="/assets/test.mp4-single-preview-1.png" width="40%"></div>

*Method 1 prompt (Single Frame - Unique Prompts)* 

#### **Method 2: Multi-Frame Prompting with Entropy Filtering**
This approach uses multiple point prompts across multiple frames to generate a single mask covering all hands. It then filters out frames with high entropy values, reducing artifacts in the mask and improving segmentation quality. Using multiple frames improves the consistency of the masks and reduces artifacts.

<div align="center"> <img src="/assets/test.mp4-multi-preview-1.png" width="40%"><img src="/assets/test.mp4-multi-preview-2.png" width="40%"></div> 

*Method 2 prompt (Multi Frame - Combined Prompt)* 

## Examples
Find the sample data and the outputs produced by both methods in the [examples](examples) folder.

## Installation & Usage

To install the package, use the following command:
```sh
pip install git+https://github.com/abhishekk962/thevisiblehand
```


### Python API Usage

```python
from thevisiblehand import VideoProcessor

# Create a processor object
processor = VideoProcessor('path/to/input.mp4', 'path/to/output.mp4')

# Preview the results of the chosen method
processor.preview_results()

# Process and save the entire masked video
processor.process_video()
```


### Command Line Interface (CLI) Usage

```sh
usage: thevisiblehand [-h] [--hands HANDS] [--method {single,multi}] [--preview]
                      input_file_path output_file_path

A tool to mask hands in videos.

positional arguments:
  input_file_path       Path to the input video file.
  output_file_path      Path to save the output video file.

options:
  -h, --help            show this help message and exit
  --hands HANDS         Number of hands in the video.
  --method {single,multi}
                        Method to use for hand detection.
  --preview             Preview the results.
```

Example usage:
```sh
thevisiblehand test.mp4 output.mp4 --method multi --preview
```

---
## **Approach**
The core approach of this project involves using MediaPipe to detect hands but only leveraging wrist positions to generate prompts for SAM2. Using all hand landmarks can lead to incorrect mask formations, concentrating around the palm instead of covering the entire hand.

<div align="center"> <img src="/assets/artifacts2.png" width="40%"></div> 

*Poor mask formation when all hand landmarks are used*

Additionally, separate and unique prompts prevent the model from generating artifacts, such as mistakenly masking the neck. 

<div align="center"><img src="/assets/artifacts1.png" width="40%"><img src="/assets/clean2.png" width="40%"></div> 

*Incorrect neck masking (left) and correct hand masking (right) using Method 1*

The project also utilizes SAM2's capability of identifying objects across multiple frames to enhance mask quality. However, multi-frame masking can introduce artifacts. To counter this, an entropy-based filtering method is applied, selecting only frames with high-quality masks.

<div align="center"><img src="/assets/artifacts1.png" width="40%"><img src="/assets/clean1.png" width="40%"></div> 

*A high-entropy mask (left) and a low-entropy mask (right) using Method 2*


---
## **Further Enhancements**
- Entropy-based filtering may face challenges if the hand landmark detector fails to detect all hands in a given frame. This can be mitigated by validating hand counts across frames.
- The two methods can be combined to improve mask quality if a reliable way to link object IDs across frames is implemented, or if certain hands remain relatively static throughout the video.

---


