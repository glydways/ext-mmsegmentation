#!/usr/bin/env python3
"""Explore model layers and find activations for visualization."""

import argparse
import warnings
from mmengine.model import revert_sync_batchnorm
from mmengine import Config, DictAction
from mmseg.apis import init_model
from mmseg.utils import register_all_modules


def explore_model_layers(config_path, checkpoint_path, device='cuda:0'):
    """Explore all layers in the model and identify activations."""

    # Initialize model
    model = init_model(config_path, checkpoint_path, device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Find all layers with activations
    activation_layers = []
    all_layers = []

    for name, module in model.named_modules():
        all_layers.append(name)

        # Check if module has activation
        if hasattr(module, 'activate') and module.activate is not None or \
            hasattr(module, 'activation') and module.activation is not None:
            activation_layers.append(name)

    print("\n" + "=" * 80)
    print("ALL LAYERS (complete list)")
    print("=" * 80)
    for i, layer in enumerate(all_layers):
        print(f"{i+1:3d}. {layer}")

    # print activation layer
    print("\n Activation layers:")
    for i, layer in enumerate(activation_layers):
        print(f"{i+1:3d}. {layer}")
    print("\n")

    print(f"Total layers: {len(all_layers)}")
    print(f"Layers with activations: {len(activation_layers)}")





def main():
    parser = argparse.ArgumentParser(description='Explore model layers and print config')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument(
        '--graph', action='store_true', help='print the models graph')
    parser.add_argument('--config', help='print the config')

    args = parser.parse_args()


    register_all_modules()


    if args.config:
        print("=" * 80)
        print("CONFIG INFORMATION")
        print("=" * 80)
        cfg = Config.fromfile(args.config)

        print(f'\n{cfg.pretty_text}')


    if args.graph:
        print("\n" + "=" * 80)
        print("MODEL GRAPH")
        print("=" * 80)
        model = init_model(args.config, device='cpu')
        print(f'Model graph:\n{str(model)}')

    # Explore model layers
    print("\n" + "=" * 80)
    print("MODEL LAYER EXPLORATION")
    print("=" * 80)
    explore_model_layers(args.config, args.checkpoint, args.device)


if __name__ == '__main__':
    main()