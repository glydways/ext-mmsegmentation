import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.engine.hooks import SegVisualizationHook

import fnmatch

@HOOKS.register_module()
class CustomSegVisualizationHook(SegVisualizationHook):
    def __init__(
            self,
            draw_filenames: Optional[Sequence[str]] = None,
            val_interval: Optional[int] = None,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.val_interval = val_interval
        self._patterns = list(draw_filenames) if draw_filenames else None

    def _draw_image(self, outputs, runner_iter):
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}'

        print(f"Drawing image: {img_path}")

        self._visualizer.add_datasample(
            window_name,
            img,
            data_sample=outputs[0],
            show=self.show,
            wait_time=self.wait_time,
            step=runner_iter)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        if self.draw is False:
            return

        # Two-level control:
        # 1. Check if this is a validation run we want to visualize
        if self.val_interval is not None:
            validation_run_number = runner.iter // self.val_interval
            # print(f"Validation runner iter: {runner.iter}, batch_idx: {batch_idx}")
            if validation_run_number % self.interval != 0:
                return

        if self._patterns is not None:
            img_path = outputs[0].img_path
            img_basename = osp.basename(img_path)

            for pat in self._patterns:
                if fnmatch.fnmatch(img_basename, pat):
                    self._draw_image(outputs, runner.iter)
                    return