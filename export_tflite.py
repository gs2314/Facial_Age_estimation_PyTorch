import argparse
import subprocess
from pathlib import Path

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model file.")
    parser.add_argument(
        "--output",
        default="age_estimator.tflite",
        help="Destination .tflite file path.",
    )
    parser.add_argument(
        "--saved-model-dir",
        default="onnx_saved_model",
        help="Temporary SavedModel output directory.",
    )
    return parser.parse_args()


def convert_onnx_to_saved_model(onnx_path, saved_model_dir):
    subprocess.run(
        [
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(saved_model_dir),
            "--output_tensor_type",
            "float32",
        ],
        check=True,
    )


def convert_saved_model_to_tflite(saved_model_dir, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    saved_model_dir = Path(args.saved_model_dir)
    output_path = Path(args.output)

    convert_onnx_to_saved_model(onnx_path, saved_model_dir)
    convert_saved_model_to_tflite(saved_model_dir, output_path)
    print(f"Saved TFLite model to: {output_path}")


if __name__ == "__main__":
    main()
