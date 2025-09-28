"""
Model Inference Module for the Defect Inspection System.

This is the core of the application, containing the computer vision
algorithm to identify, localize, and classify defects on the annular
components.
"""

import cv2
import numpy as np
import logging
from src.config import JUMP_THRESHOLD

def find_and_analyze_ring(processed_image: np.ndarray) -> dict:
    """
    Finds the annular object and analyzes its shape for defects.
    """
    _, binary_image = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    logging.debug(f"Found {len(contours)} initial contours.")

    if len(contours) < 2:
        return {"status": "Error", "defect_type": "Segmentation Failed"}

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    outer_contour, inner_contour = contours

    try:
        m_outer = cv2.moments(outer_contour)
        cx_outer = int(m_outer['m10'] / m_outer['m00'])
        cy_outer = int(m_outer['m01'] / m_outer['m00'])
        m_inner = cv2.moments(inner_contour)
        cx_inner = int(m_inner['m10'] / m_inner['m00'])
        cy_inner = int(m_inner['m01'] / m_inner['m00'])
    except ZeroDivisionError:
        return {"status": "Error", "defect_type": "Could not calculate moments"}

    center_x = int((cx_outer + cx_inner) / 2)
    center_y = int((cy_outer + cy_inner) / 2)
    logging.debug(f"Calculated robust center at: ({center_x}, {center_y})")

    for contour, name in [(outer_contour, "Outer"), (inner_contour, "Inner")]:
        radii = np.sqrt(((contour - (center_x, center_y))**2).sum(axis=2)).flatten()
        avg_radius = np.mean(radii)
        diffs = np.abs(radii - np.roll(radii, 1))
        max_jump = np.max(diffs)
        logging.debug(f"Max radial jump for {name} contour: {max_jump:.2f} pixels.")

        if max_jump > JUMP_THRESHOLD:
            logging.info(f"Defect detected on {name} contour!")
            defect_idx = np.argmax(diffs)
            defect_point = contour[defect_idx][0]
            dev_radius = radii[np.argmax(np.abs(radii - avg_radius))]
            
            defect_type = "Unknown"
            if name == "Outer":
                defect_type = "Cut" if dev_radius < avg_radius else "Flash"
            elif name == "Inner":
                defect_type = "Flash" if dev_radius < avg_radius else "Cut"

            return {
                "status": "Defective", "defect_type": defect_type,
                "location": tuple(defect_point), "center": (center_x, center_y)
            }

    return {"status": "Good", "defect_type": None, "location": None}