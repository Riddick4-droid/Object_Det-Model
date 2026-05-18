import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

class SSDHead(nn.Module):
    """
    One detection head applied to a feature map.
    It consists of two parallel 3x3 convolutions:
      - loc_head: predicts 4 deltas per anchor box
      - cls_head: predicts class scores per anchor box (including background)
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes # Fix: Store num_classes as an instance attribute
        self.loc_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        # locations: (B, A*4, H, W) -> (B, H*W*A, 4)
        loc = self.loc_head(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        # class logits: (B, A*(C+1), H, W) -> (B, H*W*A, C+1)
        cls = self.cls_head(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_anchors, self.num_classes+1)
        cls = cls.view(batch_size, -1, self.num_classes + 1)  # (B, total_anchors, C+1)
        return loc, cls

