"""
Configuration file for the Defect Inspection System.

This file centralizes key parameters and hyperparameters used in the
algorithm. This makes tuning and maintenance easier, as changes
can be made here without altering the core application logic.
"""

# --- Preprocessing Parameters ---
GAUSSIAN_BLUR_KERNEL = (5, 5)

# --- Defect Detection Parameters ---
# Defines the minimum "jump" in radius (in pixels) between adjacent
# contour points to be considered a defect.
JUMP_THRESHOLD = 2.5
