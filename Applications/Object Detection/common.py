"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        self.fpn_params['Conv_C3'] = nn.Conv2d(dummy_out_shapes[0][1][1], self.out_channels, kernel_size=1)
        self.fpn_params['Conv_C4'] = nn.Conv2d(dummy_out_shapes[1][1][1], self.out_channels, kernel_size=1)
        self.fpn_params['Conv_C5'] = nn.Conv2d(dummy_out_shapes[2][1][1], self.out_channels, kernel_size=1)

        self.fpn_params['Conv_P3'] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.fpn_params['Conv_P4'] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.fpn_params['Conv_P5'] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        strides = self.fpn_strides

        backbone_feats['c5'] = self.fpn_params['Conv_C5'](backbone_feats['c5'])
        fpn_feats['p5'] = self.fpn_params['Conv_P5'](backbone_feats['c5'])
        backbone_feats['c4'] = self.fpn_params['Conv_C4'](backbone_feats['c4'])
        backbone_feats['c4'] += F.interpolate(backbone_feats['c5'], scale_factor=int(strides['p5']/strides['p4']))
        fpn_feats['p4'] = self.fpn_params['Conv_P4'](backbone_feats['c4'])
        backbone_feats['c3'] = self.fpn_params['Conv_C3'](backbone_feats['c3'])
        backbone_feats['c3'] += F.interpolate(backbone_feats['c4'], scale_factor=int(strides['p4']/strides['p3']))
        fpn_feats['p3'] = self.fpn_params['Conv_P3'](backbone_feats['c3'])

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        B, C, H, W = shape_per_fpn_level[level_name]
        H_tensor = torch.arange(H, device=device, dtype=dtype)[:, None].repeat(1, W)
        W_tensor = torch.arange(W, device=device, dtype=dtype)[None, :].repeat(H, 1)
        H_tensor = level_stride * (H_tensor + 0.5).reshape(-1, 1)
        W_tensor = level_stride * (W_tensor + 0.5).reshape(-1, 1)
        location_coords[level_name] = torch.cat((W_tensor, H_tensor), dim=1)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    # *|CURSOR_MARCADOR|*
    N = boxes.shape[0]

    # Sort boxes based on scores
    scores_sorted = torch.sort(scores, descending=True)
    sorted_idx = scores_sorted.indices.tolist()
    keep_ls = []

    x1 = torch.maximum(boxes[:, 0][:,None].repeat(1, N), (boxes[:, 0][:, None].repeat(1,N)).t())
    y1 = torch.maximum(boxes[:, 1][:,None].repeat(1, N), (boxes[:, 1][:, None].repeat(1,N)).t())
    x2 = torch.minimum(boxes[:, 2][:,None].repeat(1, N), (boxes[:, 2][:, None].repeat(1,N)).t())
    y2 = torch.minimum(boxes[:, 3][:,None].repeat(1, N), (boxes[:, 3][:, None].repeat(1,N)).t())
    intersection = (x2 - x1) * (y2 - y1)
    intersection[x2-x1 <= 0] = 0.0
    intersection[y2-y1 <= 0] = 0.0
    area = (boxes[:, 2] - boxes[:,0]) * (boxes[:, 3] - boxes[:,1])
    union = area[:, None].repeat(1,N) + area[:, None].repeat(1,N).t() - intersection
    iou = intersection / union
    keep_ls.append(sorted_idx[0])
    
    eliminate_ls = (iou[sorted_idx[0], :] > iou_threshold).nonzero()[:,0].tolist()
    sorted_idx = [k for k in sorted_idx if k not in eliminate_ls]
    for i in sorted_idx:
      if i in eliminate_ls:
        continue
      keep_ls.append(i)
      iou_score = iou[i, :]
      eliminate_idx = (iou_score > iou_threshold).nonzero()[:, 0]
      eliminate_ls.extend(eliminate_idx.tolist())
    keep = torch.tensor(keep_ls, device=boxes.device, dtype=torch.long)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
