import torch.nn as nn
from models.basic import BasicBlock

    

class CustomBackbone(nn.Module):
    """
    This is a  tiny ResNet-like backbone that outputs feature maps at three scales:
      - scale_8: 1/8 of input spatial size
      - scale_16: 1/16 of input spatial size
      - scale_32: 1/32 of input spatial size
    Number of output channels: [64, 128, 256] (can be changed easily).
    """
    def __init__(self):
        super().__init__()
        self.in_planes = 32

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Instantiate MaxPool2d in __init__ to enable JIT scripting
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 1: 1/2 -> 1/4
        self.layer1 = self._make_layer(32, 2, stride=1)   # output size still 1/4
        # Stage 2: 1/4 -> 1/8
        self.layer2 = self._make_layer(64, 2, stride=2)   # now 1/8
        # Stage 3: 1/8 -> 1/16
        self.layer3 = self._make_layer(128, 2, stride=2)  # 1/16
        # Stage 4: 1/16 -> 1/32
        self.layer4 = self._make_layer(256, 2, stride=2)  # 1/32

        # We'll extract the outputs after layer2, layer3, layer4
        self.out_channels = [64, 128, 256]

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns a list of feature maps:
          [scale_8 (1/8), scale_16 (1/16), scale_32 (1/32)]
        """
        x = self.stem(x)          # 1/2
        x = self.maxpool(x)       # 1/4
        scale_4 = self.layer1(x)  # still 1/4
        scale_8 = self.layer2(scale_4)   # 1/8
        scale_16 = self.layer3(scale_8)  # 1/16
        scale_32 = self.layer4(scale_16) # 1/32
        return (scale_8, scale_16, scale_32)
    


