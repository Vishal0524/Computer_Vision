"""
Result Visualization Module for the Defect Inspection System.

This module handles the creation of visual outputs, overlaying the
analysis results onto the original images for user review and saving
them to a file.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

def visualize_results(image: np.ndarray, result: dict, name: str, output_dir: str):
    """
    Overlays defect information on the image and saves it to a file.
    """
    status = result["status"]
    defect_type = result["defect_type"]
    location = result["location"]

    title = "File: {}\nStatus: {}".format(name, status)
    color = 'green' if status == 'Good' else 'red'

    if status == "Defective":
        title += " - Type: {}".format(defect_type)
        logging.debug("Drawing defect marker at {}".format(location))
        cv2.circle(image, location, 20, (0, 0, 255), 3)
        cv2.line(image, result["center"], location, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title, color=color, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # Save the figure to the specified output directory
    output_filename = "result_{}".format(name)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    logging.info("Saved result image to: {}".format(output_path))
    plt.close() # Close the figure to free up memory