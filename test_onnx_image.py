import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

from config import config


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((config["img_width"], config["img_height"]))

    data = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    data = (data - mean) / std
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    return data


def main():
    parser = argparse.ArgumentParser(description="Run ONNX age estimator on a single image.")
    parser.add_argument("--model", required=True, help="Path to ONNX model file.")
    parser.add_argument("--image", required=True, help="Path to input image file.")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_data = preprocess_image(args.image)
    outputs = session.run(None, {input_name: input_data})
    age = float(outputs[0].reshape(-1)[0])
    print(f"Predicted age: {age:.2f}")


if __name__ == "__main__":
    main()
