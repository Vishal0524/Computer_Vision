"""
Logger Configuration Utility.

Provides a standardized, reusable function to set up the application's
logger. This ensures consistent log formatting across all modules.
"""

import logging

def setup_logger():
    """
    Configures the root logger for the application.
    """
    logging.basicConfig(
        level=logging.INFO,  # Change to logging.DEBUG for more verbose output
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
