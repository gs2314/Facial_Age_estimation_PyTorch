import argparse
import os

import torch
import coremltools as ct

from config import config
from model import AgeEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export age estimation model to Core ML.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--model-name",
        default=config["model_name"],
        choices=["resnet", "efficientnet", "vit"],
        help="Backbone architecture to export.",
    )
    parser.add_argument(
        "--output",
        default="age_estimator.mlmodel",
        help="Destination .mlmodel file path.",
    )
    return parser.parse_args()


def export_coreml(checkpoint_path, model_name, output_path):
    model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name=model_name,
        pretrain_weights=None,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        config["img_height"],
        config["img_width"],
        dtype=torch.float32,
    )
    traced = torch.jit.trace(model, dummy_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",
                shape=dummy_input.shape,
            )
        ],
        convert_to="mlprogram",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)


if __name__ == "__main__":
    args = parse_args()
    export_coreml(args.checkpoint, args.model_name, args.output)
