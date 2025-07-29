import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.engine.hooks import SegVisualizationHook


@HOOKS.register_module()
class CustomSegVisualizationHook(SegVisualizationHook):
    def __init__(self, draw_idx=None, draw_filenames=None, *args, **kwargs):
        self.draw_idx = draw_idx
        self.draw_filenames = draw_filenames
        super().__init__(*args, **kwargs)

    def _draw_image(self, outputs, total_curr_iter):
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
            step=total_curr_iter)


    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        if self.draw is False:
            return

        total_curr_iter = runner.iter + batch_idx # total_curr_iter is the total number of iterations (including the current iteration)
        if total_curr_iter % self.interval != 0:
            return

        if self.draw_idx is not None and batch_idx == self.draw_idx:
            self._draw_image(outputs, total_curr_iter)

        if self.draw_filenames is not None:
            img_path = outputs[0].img_path
            img_basename = osp.basename(img_path)

            # Check if filename matches any in draw_filenames
            # Handle both formats: with and without _img suffix
            for filename in self.draw_filenames:
                # Direct match
                if img_basename == filename:
                    self._draw_image(outputs, total_curr_iter)
                    return
                # Match without _img suffix
                if img_basename == filename.replace('.png', '_img.png'):
                    self._draw_image(outputs, total_curr_iter)
                    return
                # Match with _img suffix
                if img_basename == filename.replace('.png', '_img.png'):
                    self._draw_image(outputs, total_curr_iter)
                    return




