from pathlib import Path

import onnx
import torch
from ultralytics import YOLO

import max_benchmark

MODEL_DIR = Path(max_benchmark.__file__).parent / "../../models/ultralytics"

def load_torchscript(model_name, batch_size):
    model_path = MODEL_DIR / f"{model_name}.torchscript"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    size = (480, 640)
    input_batch = torch.zeros((batch_size, 3, size[0], size[1]), dtype=torch.float32)

    if model_path.exists():
        return model_path, input_batch.numpy()

    model = YOLO(f"{MODEL_DIR}/{model_name}.pt")
    model.export(format="torchscript", imgsz=size)

    assert model_path.exists()
    return model_path, input_batch.numpy()


def load_onnx(model_name, batch_size):
    model_path = MODEL_DIR / f"{model_name}.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    size = (480, 640)
    input_batch = torch.zeros((batch_size, 3, size[0], size[1]), dtype=torch.float32)

    if model_path.exists():
        return model_path, input_batch.numpy()

    model = YOLO(f"{MODEL_DIR}/{model_name}.pt")
    model.export(format="onnx", imgsz=size, simplify=True)

    rename_onnx_input(model_path, "images", "input")

    assert model_path.exists()
    return model_path, input_batch.numpy()


def rename_onnx_input(model_path, old_name, new_name):
    model = onnx.load(model_path)
    for node in model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] == old_name:
                node.input[i] = new_name

    for input in model.graph.input:
        if input.name == old_name:
            input.name = new_name

    onnx.save(model, model_path)
