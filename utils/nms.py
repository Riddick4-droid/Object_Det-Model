import torch
from utils.logger import get_logger
from torchvision.ops import nms

logger = get_logger()


#non max suppression utility function
def non_max_suppression(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on a set of bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of bounding boxes of shape (N, 4),
                              where N is the number of boxes, and each box
                              is represented as [x1, y1, x2, y2].
        scores (torch.Tensor): A tensor of confidence scores for each bounding box (N,).
        iou_threshold (float): The Intersection Over Union (IOU) threshold for suppression.

    Returns:
        torch.Tensor: A tensor of indices of the boxes to keep after NMS.
    """
    global logger

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)

    if boxes.numel() == 0 or scores.numel() == 0:
        logger.warning("No boxes or scores provided for NMS. Returning empty tensor.")
        return torch.tensor([], dtype=torch.long)

    # torchvision.ops.nms expects boxes in (x1, y1, x2, y2) format
    keep_indices = nms(boxes, scores, iou_threshold)
    logger.info(f"NMS completed. Kept {len(keep_indices)} out of {len(boxes)} boxes.")
    return keep_indices

if __name__ == "__main__":
    logger.info('NMS function created successfully!!')