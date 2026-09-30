
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.embed import PatchEmbed, PatchMerging


@BACKBONES.register_module()
class Self_Define_Backbone(BaseModule):
    def __init__(self, in_channels=3):
        self.in_channels = in_channels

    def forward(self, x):
        return x, x, x, x
