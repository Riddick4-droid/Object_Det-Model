import json
from PIL import Image
from typing import List, Dict, Tuple, TypedDict
import os
from utils.logger import get_logger
from utils.display_images import AnnotatedImage

logger = get_logger()

# Define a TypedDict for better structure and type hints for annotations
# this acts like a schema to support the load_image_annotations


def load_image_annotations(image_folder_path: str,
                           annotations_file_path: str) -> List[AnnotatedImage]:
    """
    Loads images and their corresponding annotations (bounding boxes and labels).

    Args:
        image_folder_path (str): Path to the directory containing the images.
        annotations_file_path (str): Path to the JSON file containing annotations.

    Returns:
        List[AnnotatedImage]: A list of dictionaries, where each dictionary contains
                               image path, the PIL Image object, bounding boxes, and labels.
    """
    global logger

    annotations_list: List[AnnotatedImage] = []

    if not os.path.exists(image_folder_path):
        logger.error(f"Image folder not found: {image_folder_path}")
        return []

    if not os.path.exists(annotations_file_path):
        logger.error(f"Annotations file not found: {annotations_file_path}")
        return []

    try:
        with open(annotations_file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {annotations_file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading annotations file {annotations_file_path}: {e}")
        return []

    # the JSON structure has an 'annotations' key and each annotation has 'image_id', 'bbox', 'category_id'
    # And an 'images' key with 'id', 'file_name'
    # And a 'categories' key with 'id', 'name'

    images_meta = {img['id']: img['file_name'] for img in data.get('images', [])}
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # Group annotations by image_id
    grouped_annotations: Dict[int, List[Dict]] = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in grouped_annotations:
            grouped_annotations[img_id] = []
        grouped_annotations[img_id].append(ann)

    for img_id, anns in grouped_annotations.items():
        file_name = images_meta.get(img_id)
        if not file_name:
            logger.warning(f"Image ID {img_id} has annotations but no corresponding file name in 'images' metadata. Skipping.")
            continue

        image_path = os.path.join(image_folder_path, file_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}. Skipping annotations for this image.")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}. Skipping.")
            continue

        boxes: List[List[float]] = []
        labels: List[str] = []

        for ann in anns:
            bbox = ann['bbox'] # [x, y, width, height] format
            category_id = ann['category_id']

            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])

            label_name = categories.get(category_id, 'unknown')
            labels.append(label_name)

        annotations_list.append({
            'image_path': image_path,
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'file_name': file_name
        })
    logger.info(f"Successfully loaded {len(annotations_list)} images with annotations.")
    return annotations_list