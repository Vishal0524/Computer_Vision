"""
Main entry point for the Automated Defect Inspection System.

This script orchestrates the entire pipeline:
1. Sets up the application logger.
2. Specifies the image data to be processed.
3. Iterates through each image, passing it to the relevant modules for
   preprocessing, analysis, and visualization.

Author: Vishal R
Date: September 27, 2025
"""

import os
import logging
from src.data_loader import preprocess_image
from src.defect_detector import find_and_analyze_ring
from src.visualizer import visualize_results
from utils.logger_config import setup_logger

def run_inspection_pipeline():
    """
    Executes the full defect detection and analysis pipeline.
    """
    setup_logger()
    logging.info("--- Starting Automated Defect Inspection ---")

    data_dir = "data"
    # Define an output directory for saving results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Output will be saved to the '{}' directory.".format(output_dir))

    image_files = [
        "defect1.png",
        "defect2.png",
        "defect3.png",
        "defect4.png",
        "good.png",
    ]

    for file_name in image_files:
        image_path = os.path.join(data_dir, file_name)
        logging.info("Processing image: {}".format(image_path))

        # Step A: Preprocess the image
        original_img, processed_img = preprocess_image(image_path)
        if original_img is None:
            continue

        # Step B: Run inference to find defects
        result = find_and_analyze_ring(processed_img)
        logging.info("Analysis result: {}".format(result['status']))
        if result['status'] == 'Defective':
            logging.info("Detected Defect Type: {}".format(result['defect_type']))
        elif result['status'] == 'Error':
            logging.error("An error occurred: {}".format(result['defect_type']))

        # Step C: Visualize the output
        visualize_results(original_img, result, file_name, output_dir)

    logging.info("--- Inspection Complete ---")

if __name__ == "__main__":
    run_inspection_pipeline()