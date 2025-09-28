"""
Data Preparation Module for the Defect Inspection System.

This module is responsible for all data loading and preprocessing tasks.
It ensures that images are loaded correctly and prepared in a consistent
format for the analysis module.
"""

import cv2
import logging
import numpy as np
from src.config import GAUSSIAN_BLUR_KERNEL

def preprocess_image(image_path: str):
    """
    Loads and preprocesses the input image.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        logging.error(f"Could not load image at path: {image_path}")
        return None, None

    logging.debug(f"Successfully loaded image: {image_path}")
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_BLUR_KERNEL, 0)
    logging.debug("Image converted to grayscale and blurred.")

    return original_image, blurred_image