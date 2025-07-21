from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class GlydLensArtifactDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'raindrop'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='_img.png',
                 seg_map_suffix='_label.png', # TODO
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)