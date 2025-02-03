import argparse
from thevisiblehand import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="A tool to mask hands in videos.")
    parser.add_argument("input_file_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_file_path", type=str, help="Path to save the output video file.")
    parser.add_argument("--hands", type=int, default=2, help="Number of hands in the video.")
    parser.add_argument("--method", type=str, choices=["single", "multi"], default="single", help="Method to use for hand detection.")
    parser.add_argument("--preview", action="store_true", help="Preview the results.")

    args = parser.parse_args()

    processor = VideoProcessor(args.input_file_path, args.output_file_path, args.num_hands, args.method)
    try:
        if args.preview:
            processor.preview_results()
        else:
            processor.process_video()
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()