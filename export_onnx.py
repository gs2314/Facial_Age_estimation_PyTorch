import argparse
import os

import torch

from config import config
from model import AgeEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export age estimation model to ONNX.")
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
        default="age_estimator.onnx",
        help="Destination .onnx file path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def export_onnx(checkpoint_path, model_name, output_path, opset):
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["age"],
        dynamic_axes={"input": {0: "batch"}, "age": {0: "batch"}},
        opset_version=opset,
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.checkpoint, args.model_name, args.output, args.opset)
