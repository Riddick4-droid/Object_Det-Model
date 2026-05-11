import random
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from utils.logger import get_logger
from typing import TypedDict, List
from torchvision import models
from torchvision.transforms import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms
import random
from torchvision.transforms import ToTensor
from models.model import model
from configs.config import DEVICE



logger = get_logger()


#schema for annotated images
class AnnotatedImage(TypedDict):
    image_path: str
    image: Image.Image
    boxes: List[List[float]]  # List of [x_min, y_min, x_max, y_max]/ that is a list of list of box coordinates
    labels: List[str]
    file_name: str


##utility function to display and createa grid of 6 images randomly from the train dataset

def display_and_create_grid(dataset, padding:int = 5, 
                            num_images=6, 
                            save_grid:bool=False, 
                            normalize:bool=False):
    global logger

    if not dataset:
        logger.warning("Dataset is empty or None. Cannot display images.")
        return

    if len(dataset) < num_images:
        logger.warning(f"Dataset has only {len(dataset)} images, displaying all of them instead of {num_images}.")
        num_images = len(dataset)

    # Select random images
    random_indices = random.sample(range(len(dataset)), num_images)
    selected_images = []

    # Retrieve images and convert to tensor if necessary
    # ImageFolder dataset returns (image, label), so we take only the image
    for i in random_indices:
        image, _ = dataset[i]
        if isinstance(image, Image.Image):
            # Convert PIL Image to tensor for make_grid
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        selected_images.append(image)

    if not selected_images:
        logger.warning("No images were selected for display.")
        return

    # create a grid of images
    grid = make_grid(selected_images,
                     nrow=int(num_images**0.5),
                     padding=padding,
                     normalize=normalize,
                     scale_each=True)

    # Convert the grid tensor to a PIL Image or numpy array for matplotlib
    # make_grid outputs a tensor of shape (C, H, W). For matplotlib, we need (H, W, C) for color images.
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Display the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(ndarr)
    plt.axis('off')
    plt.title(f'{num_images} Random Images from Dataset')

    if save_grid:
        try:
            os.makedirs('image_grids', exist_ok=True)
            filename = f'image_grids/random_grid_{num_images}.png'
            plt.savefig(filename)
            logger.info(f"Image grid saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving image grid: {e}")

    plt.show()




def load_and_display_annotated_images(
    annotated_images: List[AnnotatedImage],
    num_images_to_display: int = 5,
    display: bool = True
) -> List[torch.Tensor]:
    """
    Loads a specified number of random images from a list of annotated images,
    draws bounding boxes on them, and optionally displays them.

    Args:
        annotated_images (List[AnnotatedImage]): A list of AnnotatedImage dictionaries.
        num_images_to_display (int): The number of images to randomly select and display.
        display (bool): If True, the images will be displayed using matplotlib.

    Returns:
        List[torch.Tensor]: A list of PyTorch Tensors of the images with drawn bounding boxes.
    """
    logger = get_logger()
    if not annotated_images:
        logger.warning("No annotated images provided.")
        return []

    if num_images_to_display > len(annotated_images):
        logger.warning(
            f"Requested {num_images_to_display} images, but only {len(annotated_images)} are available. "
            "Displaying all available images."
        )
        num_images_to_display = len(annotated_images)

    # Randomly select images to display
    random_indices = random.sample(range(len(annotated_images)), num_images_to_display)
    images_with_boxes = []
    to_tensor = ToTensor()

    for idx in random_indices:
        ann_data = annotated_images[idx]
        image_pil = ann_data['image']
        boxes = ann_data['boxes']
        labels = ann_data['labels']
        file_name = ann_data['file_name']

        # Convert PIL Image to a PyTorch Tensor
        image_tensor = to_tensor(image_pil)
        # Convert to uint8 for draw_bounding_boxes function
        image_tensor = (image_tensor * 255).to(torch.uint8)

        if boxes:
            # Convert boxes to tensor (draw_bounding_boxes expects tensor)
            # Ensure boxes are in (xmin, ymin, xmax, ymax) and are integers
            boxes_tensor = torch.tensor(boxes, dtype=torch.float)
            # Optionally, you can pass labels if you want them drawn on the image
            # For simplicity, we are just drawing boxes here
            annotated_image = draw_bounding_boxes(
                image_tensor,
                boxes_tensor,
                labels=labels,
                colors="red",
                width=2
            )
        else:
            annotated_image = image_tensor

        images_with_boxes.append(annotated_image)
        logger.info(f"Processed image: {file_name} with {len(boxes)} boxes.")

    if display and images_with_boxes:
        grid = make_grid(images_with_boxes, nrow=min(num_images_to_display, 4), padding=5)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(15, 15))
        plt.imshow(ndarr)
        plt.title(f"Random {num_images_to_display} Annotated Images")
        plt.axis('off')
        plt.show()

    return images_with_boxes

import random
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes, make_grid

# Define the transformation for input images to the model
# Assuming the model expects normalized tensors
transform = transforms.Compose([
    transforms.Resize((model.input_size, model.input_size)), # Resize to model's expected input size
    transforms.ToTensor(),
    # ypu can add normalization if the model was trained with it; add it like this below:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#function to display model predictions
def display_model_predictions(
    model,
    annotated_images: List[AnnotatedImage],
    class_names: List[str],
    num_images_to_display: int = 3,
    confidence_threshold: float = 0.5,
    show_ground_truth: bool = False,
    apply_nms: bool = False, # parameter to apply NMS(Non MAx Suppression)
    nms_iou_threshold: float = 0.5 # parameter for NMS IOU threshold
):
    logger = get_logger()
    if not annotated_images:
        logger.warning("No annotated images provided.")
        return

    if num_images_to_display > len(annotated_images):
        num_images_to_display = len(annotated_images)

    random_indices = random.sample(range(len(annotated_images)), num_images_to_display)
    predicted_images = []

    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # this disables gradient calculations. same as torch.inference_mode()
        for idx in random_indices:
            ann_data = annotated_images[idx]
            image_pil = ann_data['image']
            file_name = ann_data['file_name']

            #1. Preprocess the image for the model
            image_tensor = transform(image_pil).to(device)
            # Add batch dimension
            image_batch = image_tensor.unsqueeze(0)

            #2. Get model predictions
            predictions = model(image_batch)

            #3. Extract and filter raw predictions based on confidence
            pred_boxes = predictions['boxes'][0] # Take first item from batch
            pred_labels = predictions['labels'][0]
            pred_scores = predictions['scores'][0]

            # Initial filtering by confidence threshold
            keep_preds_conf = pred_scores > confidence_threshold
            filtered_pred_boxes = pred_boxes[keep_preds_conf]
            filtered_pred_labels_raw = pred_labels[keep_preds_conf]
            filtered_pred_scores = pred_scores[keep_preds_conf]

            # If no boxes passed the confidence threshold AND threshold is 0.0 (for debugging untrained models),
            # show a few top-scoring predictions anyway to ensure some visualization.
            if filtered_pred_boxes.numel() == 0 and confidence_threshold == 0.0:
                logger.info(f"No predictions passed the 0.0 confidence threshold for {file_name}. Displaying top 100 highest scoring predictions for debugging visualization.")
                top_k_to_show = min(100, len(pred_scores)) # Limit to 100 boxes, or fewer if not many predictions
                if top_k_to_show > 0:
                    # Get top N scores, even if very low, to ensure *some* boxes are drawn.
                    top_scores, top_indices = torch.topk(pred_scores, top_k_to_show, largest=True)
                    filtered_pred_boxes = pred_boxes[top_indices]
                    filtered_pred_labels_raw = pred_labels[top_indices]
                    filtered_pred_scores = pred_scores[top_indices]
                else:
                    logger.warning(f"No predictions generated by the model for {file_name}.")

            # Apply NMS if needed, set to true
            if apply_nms and filtered_pred_boxes.numel() > 0:
                # NMS expects boxes in (x1, y1, x2, y2) and scores
                nms_keep_indices = non_max_suppression(
                    boxes=filtered_pred_boxes,
                    scores=filtered_pred_scores,
                    iou_threshold=nms_iou_threshold
                )
                #keep only indices where the nms is applied and chosen
                filtered_pred_boxes = filtered_pred_boxes[nms_keep_indices]
                filtered_pred_labels_raw = filtered_pred_labels_raw[nms_keep_indices]
                filtered_pred_scores = filtered_pred_scores[nms_keep_indices]
                logger.info(f"NMS applied to predictions for {file_name}. Kept {len(filtered_pred_boxes)} boxes.")

            # Prepare original image for drawing (uint8, C, H, W)
            # The image_tensor here is ALREADY resized to model.input_size
            image_to_draw = (image_tensor * 255).to(torch.uint8)

            all_boxes_to_draw = []
            all_labels_to_display = []
            all_colors = []

            # Add Ground Truth to show how well model performed in drawing boxes around the object compared to the actual
            if show_ground_truth:
                gt_boxes_raw = ann_data['boxes']
                gt_labels_raw = ann_data['labels']
                if gt_boxes_raw:
                    # Get original image dimensions from PIL image before transformation
                    original_width, original_height = image_pil.size

                    # Calculate scaling factors
                    scale_x = model.input_size / original_width
                    scale_y = model.input_size / original_height

                    # Scale ground truth boxes
                    scaled_gt_boxes = []
                    for bbox in gt_boxes_raw:
                        # Correct unpacking of bbox (it's already xmin, ymin, xmax, ymax)
                        x_min, y_min, x_max, y_max = bbox
                        scaled_gt_boxes.append([
                            x_min * scale_x,
                            y_min * scale_y,
                            x_max * scale_x,
                            y_max * scale_y
                        ])

                    # Move ground truth boxes to the same device as model predictions
                    gt_boxes_tensor = torch.tensor(scaled_gt_boxes, dtype=torch.float32).to(device)
                    all_boxes_to_draw.append(gt_boxes_tensor)
                    all_labels_to_display.extend([f"GT: {lbl}" for lbl in gt_labels_raw])
                    all_colors.extend(["red"] * len(gt_boxes_raw))
                logger.info(f"Processed ground truth for {file_name}. Detected {len(gt_boxes_raw)} objects.")


            # ===Add Model Predictions ===
            if filtered_pred_boxes.numel() > 0:
                pred_labels_text = []
                pred_boxes_for_display = []

                for label, score, box in zip(filtered_pred_labels_raw, filtered_pred_scores, filtered_pred_boxes):
                    # Filter out background predictions if confidence_threshold is not 0.0
                    # Background class has label 0.
                    if label.item() == 0 and confidence_threshold != 0.0:
                        continue

                    pred_boxes_for_display.append(box)
                    if label.item() == 0:
                        pred_labels_text.append(f"Pred: Background: {score.item():.2f}")
                    else:
                        # Ensure class_names index is valid (label.item() is 1-indexed for actual classes)
                        pred_labels_text.append(f"Pred: {class_names[label.item()-1]}: {score.item():.2f}")

                if pred_boxes_for_display:
                    all_boxes_to_draw.append(torch.stack(pred_boxes_for_display))
                    all_labels_to_display.extend(pred_labels_text)
                    all_colors.extend(["green"] * len(pred_boxes_for_display))
                    logger.info(f"Processed predictions for {file_name}. Displaying {len(pred_boxes_for_display)} objects (excluding background if threshold > 0). Total detected by model: {len(filtered_pred_boxes)}.")
                else:
                    logger.info(f"No relevant (non-background or above threshold) predictions to display for {file_name}.")


            final_drawn_image = image_to_draw
            if all_boxes_to_draw:
                combined_boxes_tensor = torch.cat(all_boxes_to_draw, dim=0)
                final_drawn_image = draw_bounding_boxes(
                    image_to_draw,
                    combined_boxes_tensor,
                    labels=all_labels_to_display,
                    colors=all_colors,
                    width=2
                )
            predicted_images.append(final_drawn_image)

    if predicted_images:
        grid = make_grid(predicted_images, nrow=num_images_to_display, padding=5)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(15, 15))
        plt.imshow(ndarr)
        title_text = ""
        if show_ground_truth:
            title_text += "Ground Truth (Red) & "
        title_text += f"Model Predictions (Green, Confidence > {confidence_threshold})"
        if apply_nms:
            title_text += f" with NMS (IoU > {nms_iou_threshold})"
        plt.title(title_text)
        plt.axis('off')
        plt.show()
