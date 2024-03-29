from .two_stage_rbbox import TwoStageDetectorRbbox
from ..registry import DETECTORS

from mmdet.models.builder import DETECTORS
@DETECTORS.register_module
class FasterRCNNOBB(TwoStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 rpn_head,

                 train_cfg,
                 test_cfg,
                 roi_head=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNNOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            roi_head=roi_head,
            # bbox_roi_extractor=bbox_roi_extractor,
            # bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
