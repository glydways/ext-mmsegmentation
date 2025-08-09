from argparse import ArgumentParser
from typing import Type

import mmcv
import torch
import torch.nn as nn

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer

# Example usage: python visualize_feature_map.py \
#   /path/to/your/image.jpg \
#   /path/to/your/config.py \
#   /path/to/your/checkpoint.pth \
#   --gt_mask /path/to/your/gt_mask.png \
#   --device cuda:0 \
#   --opacity 0.5 \
#   --title "result" \
#   --layers "backbone.spatial_path.layer1 backbone.context_path.backbone.layer1.1 backbone.context_path.arm16"

class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, module: nn.Module, input: Type, output: Type):
        print(f"HOOK FIRED: {module} (recording feature shape {output.shape})")
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result, layer_names):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    image = mmcv.imread(args.img, 'color')

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)
    # pprint(len(recorder.data_buffer))
    # print("====================================")
    # add feature map to wandb visualizer
    for idx, layer_name in enumerate(layer_names):
        # print(recorder.data_buffer[idx].shape)
        feature = recorder.data_buffer[idx][0]  # assuming batch dim
        drawn = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='squeeze_mean'
        )
        # print(drawn.shape)
        seg_visualizer.add_image(f'feature_{layer_name}', drawn)
        # print("====================================")

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument(
        '--layers',
        nargs='+',
        default=None,
        help='Specific layer names to visualize (e.g., backbone.spatial_path.layer1 backbone.context_path.arm16.conv_layer)')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Determine which layers to visualize, double check the model architecture. Get the layer names using explore_model_layers.py
    if args.layers:
        source = args.layers
    else:
        # Default layers for BiSeNetV1
        source = [
            'backbone.spatial_path.layer1.activate',
            'backbone.spatial_path.layer2.activate',
            'backbone.spatial_path.layer3.activate',
            'backbone.spatial_path.layer4.activate',
            'backbone.context_path.backbone.layer1.1',
            'backbone.context_path.backbone.layer2.1',
            'backbone.context_path.backbone.layer3.1',
            'backbone.context_path.backbone.layer4.1',
            'backbone.context_path.arm16',
            'backbone.context_path.arm32',
            'backbone.ffm',
            'decode_head.convs.0.activate',
            'decode_head.conv_seg'
        ]

    source = dict.fromkeys(source)

    recorder = Recorder()
    layer_names = []  # Store layer names in order

    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            layer_names.append(name)
            handle = module.register_forward_hook(recorder.record_data_hook)
            # recorder._handles.append(handle)
    print(layer_names)
    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)

    visualize(args, model, recorder, result, layer_names)


if __name__ == '__main__':
    main()
