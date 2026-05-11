#final model
import torch.nn as nn
import torch
import math
from models.backbone import CustomBackbone
from models.head import SSDHead
from torch.functional import F

class ObjDet_V1(nn.Module):
    """
    Custom object detector from scratch.
    Backbone: our tiny ResNet variant (multi‑scale outputs).
    Heads: SSD style box and class predictors.
    The forward pass returns decoded boxes, class labels, and confidence scores.
    """
    def __init__(self, num_classes, input_size=512):
        """
        Args:
            num_classes: number of object classes (does not include the background).
            input_size: assumed square image size (used only for anchor generation).
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone = CustomBackbone()
        out_channels = self.backbone.out_channels  # [64, 128, 256]

        # Anchor configuration per scale:
        # Each dict defines: feature map stride, sizes (in pixels) and aspect ratios.
        self.feature_maps = [
            {"stride": 8,  "sizes": [30, 60],   "ratios": [1, 2, 0.5]},                     # scale_8
            {"stride": 16, "sizes": [90, 120],  "ratios": [1, 2, 0.5, 3, 0.33]},            # scale_16
            {"stride": 32, "sizes": [150, 200], "ratios": [1, 2, 0.5]}                      # scale_32
        ]

        # Build heads for each scale
        self.heads = nn.ModuleList()
        self.num_anchors_per_loc = []
        for fm, in_ch in zip(self.feature_maps, out_channels):
            num_anchors = len(fm["sizes"]) * len(fm["ratios"])
            self.num_anchors_per_loc.append(num_anchors)
            self.heads.append(SSDHead(in_ch, num_anchors, num_classes))

        # Pre‑compute all anchor boxes (image coordinates) and store as buffer
        self.input_size = input_size
        anchors = self._generate_anchors()
        self.register_buffer("anchors", anchors)  # shape (total_anchors, 4)

    def _generate_anchors(self):
        """
        Build anchor boxes for every feature map location according to the
        defined strides, sizes and ratios. Coordinates are in (x1, y1, x2, y2)
        relative to the full input image (0..input_size).
        """
        all_anchors = []
        for fm in self.feature_maps:
            stride = fm["stride"]
            sizes = fm["sizes"]
            ratios = fm["ratios"]
            fm_h = fm_w = self.input_size // stride
            # Pre‑compute box templates (half width & height) for [x1,y1,x2,y2]
            boxes = []
            for s in sizes:
                for r in ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    # box in (cx, cy, w, h) -> (x1,y1,x2,y2) with (0,0) center
                    boxes.append([-w/2, -h/2, w/2, h/2])
            boxes = torch.tensor(boxes, dtype=torch.float32)  # (A, 4) in cx-centered half-sizes

            # Create grid of centres
            grid_y, grid_x = torch.meshgrid(
                torch.arange(fm_h, dtype=torch.float32),
                torch.arange(fm_w, dtype=torch.float32),
                indexing='ij'
            )
            # centre coordinates in image pixels (0.5 offset to pixel centre)
            cy = (grid_y + 0.5) * stride
            cx = (grid_x + 0.5) * stride
            cy = cy.reshape(-1, 1)  # (H*W, 1)
            cx = cx.reshape(-1, 1)
            centres = torch.cat([cx, cy, cx, cy], dim=1)  # (H*W, 4) to add to boxes

            # Broadcast: (H*W, A, 4)
            anchors_fm = centres.unsqueeze(1) + boxes.unsqueeze(0)  # (H*W, A, 4)
            anchors_fm = anchors_fm.reshape(-1, 4)  # (H*W*A, 4)
            all_anchors.append(anchors_fm)

        return torch.cat(all_anchors, dim=0)  # (total_anchors, 4)

    def _decode_boxes(self, loc_preds):
        """
        Decode raw bounding box predictions into (x1,y1,x2,y2) image coordinates.
        loc_preds: (B, total_anchors, 4) = predicted offsets in SSD style
                    (delta_cx, delta_cy, delta_w, delta_h) w.r.t. anchor centre and size.
        """
        anchors = self.anchors  # (N, 4)
        # Convert anchor from (x1,y1,x2,y2) to (cx, cy, w, h)
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

        # Predicted deltas
        dx = loc_preds[:, :, 0]
        dy = loc_preds[:, :, 1]
        dw = loc_preds[:, :, 2]
        dh = loc_preds[:, :, 3]

        # Decode
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h

        # Back to (x1,y1,x2,y2)
        x1 = pred_cx - pred_w / 2.0
        y1 = pred_cy - pred_h / 2.0
        x2 = pred_cx + pred_w / 2.0
        y2 = pred_cy + pred_h / 2.0

        return torch.stack([x1, y1, x2, y2], dim=2)  # (B, N, 4)

    def forward(self, images):
        """
        images: (B, 3, H, W) with H=W=self.input_size ideally.

        Returns dict:
            "boxes":  decoded bounding boxes in (x1,y1,x2,y2) format (pixel coords).
            "labels": predicted class index (0 background, 1..C = object classes).
            "scores": confidence score for the predicted class (max probability after softmax).
            If in training mode, also returns raw 'loc_preds' and 'cls_logits' for loss calculation.
        """
        features = self.backbone(images)  # list of [feat_8, feat_16, feat_32]

        locs, clss = [], []
        for feat, head in zip(features, self.heads):
            l, c = head(feat)  # (B, H*W*A, 4), (B, H*W*A, C+1)
            locs.append(l)
            clss.append(c)

        # Concatenate all anchors from all scales
        loc_preds = torch.cat(locs, dim=1)   # (B, total_anchors, 4)
        cls_logits = torch.cat(clss, dim=1)  # (B, total_anchors, C+1)

        # Decode boxes
        decoded_boxes = self._decode_boxes(loc_preds)

        # Softmax to get probabilities, then pick top class and score
        probs = F.softmax(cls_logits, dim=-1)  # (B, N, C+1)
        scores, labels = torch.max(probs, dim=-1)  # (B, N)

        output = {
            "boxes": decoded_boxes,
            "labels": labels,
            "scores": scores,
            "loc_preds": loc_preds,  # Always return raw predictions
            "cls_logits": cls_logits # Always return raw predictions
        }
        return output