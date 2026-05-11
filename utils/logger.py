import logging
import os

#setup logging utility function
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # Check if handlers are not already set
        handler = logging.StreamHandler()
        # Include %(name)s in the formatter for clearer output
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

#use this so that you can ran the code directly
if __name__ == "__main__":
    logger = get_logger()
    print(f"[INFO]: logger function enabled successfully!!")