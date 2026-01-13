import argparse
import os

import torch
import coremltools as ct

from config import config
from model import AgeEstimationModel


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.model = model

    def forward(self, x):
        return self.model(self.normalize(x))


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
    parser.add_argument(
        "--image-input",
        action="store_true",
        help="Export with Core ML image input and embed normalization.",
    )
    return parser.parse_args()


def export_coreml(checkpoint_path, model_name, output_path, image_input):
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
    if image_input:
        wrapped = ModelWithNormalize(model, config["mean"], config["std"])
        traced = torch.jit.trace(wrapped, dummy_input)
        inputs = [
            ct.ImageType(
                name="input",
                shape=dummy_input.shape,
                scale=1 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ]
    else:
        traced = torch.jit.trace(model, dummy_input)
        inputs = [
            ct.TensorType(
                name="input",
                shape=dummy_input.shape,
            )
        ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)


if __name__ == "__main__":
    args = parse_args()
    export_coreml(args.checkpoint, args.model_name, args.output, args.image_input)
