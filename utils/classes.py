from utils.display_images import AnnotatedImage
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger()

def get_num_classes(annotations: List[AnnotatedImage]) -> Tuple[int, List[str]]:
    """
    Calculates the number of unique classes from annotations.

    Args:
        annotations (List[AnnotatedImage]): A list of AnnotatedImage dictionaries.

    Returns:
        Tuple[int, List[str]]: A tuple containing the total number of classes
                               (including background) and a sorted list of class names.
    """
    global logger 
    all_labels = set() #create a set and ensure no repetitions
    for ann_data in annotations:
        for label in ann_data['labels']:
            all_labels.add(label)

    # Sort the labels for consistent mapping if needed later
    class_names = sorted(list(all_labels))
    # Add 1 for the background class, which is standard for most object detection models
    num_classes = len(class_names) + 1

    logger.info(f"Detected classes: {class_names}")
    logger.info(f"Total number of classes (including background): {num_classes}")
    return num_classes, class_names