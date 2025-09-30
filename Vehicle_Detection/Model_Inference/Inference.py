"""
YOLO inference script to process images and measure performance.

This script runs a YOLO model on a directory of images, saves the annotated
results, and calculates the inference speed in Frames Per Second (FPS).
"""

import sys
import time
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


def get_image_paths(input_path: Path) -> List[Path]:
    """Finds all image files in a given directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.glob(ext))
    return image_paths


def measure_performance(model_path: str, input_path: str, output_dir: str = None):
    """
    Run YOLO inference on image(s) and calculate performance.

    Args:
        model_path (str): Path to the trained .pt model file.
        input_path (str): Path to a single image or a folder of images.
        output_dir (str, optional): Directory to save results.
                                    Defaults to "output".
    """
    # --- 1. Setup Paths ---
    source_path = Path(input_path)
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(f"output_{source_path.stem}")
    output_path.mkdir(exist_ok=True)

    # --- 2. Load Model and Images ---
    try:
        model = YOLO(model_path)
        images = [source_path] if source_path.is_file() else get_image_paths(source_path)

        if not images:
            print(f"Error: No images found in '{input_path}'.")
            return
    except Exception as e:
        print(f"Error during setup: {e}")
        return

    print(f"Loaded model: {model_path}")
    print(f"Found {len(images)} image(s) in '{input_path}'.")
    print("-" * 30)

    # --- 3. Run Inference and Time It ---
    processed_count = 0
    start_time = time.monotonic()

    try:
        for i, img_path in enumerate(images, 1):
            results = model(str(img_path), verbose=False)[0]
            annotated_img = results.plot()
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), annotated_img)
            processed_count = i
            print(f"\rProcessing: [{i}/{len(images)}] {img_path.name}", end="")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        end_time = time.monotonic()

    # --- 4. Calculate and Display Performance ---
    total_time = end_time - start_time
    print(f"\n" + "-" * 30)
    print("Inference Complete.")

    if processed_count > 0 and total_time > 0:
        fps = processed_count / total_time
        print(f"\n--- Performance Summary ---")
        print(f"Processed Images: {processed_count}")
        print(f"Total Time Taken: {total_time:.2f} seconds")
        print(f"Inference Speed:  {fps:.2f} FPS")
        print(f"---------------------------")
    else:
        print("No images were processed to calculate performance.")

    print(f"Results saved in: {output_path.resolve()}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python your_script_name.py <path_to_model.pt> <path_to_image_or_folder> [output_directory]")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    measure_performance(model_path, input_path, output_dir)


if __name__ == "__main__":
    main()

    
