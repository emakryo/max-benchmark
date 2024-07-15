import time
from pathlib import Path

import torch 
import onnxruntime as ort
from max import engine
from torchvision.models import get_model


def load_torchscript(model_name):
    model_name = "mobilenet_v3_large"
    model_path = Path(__file__).parent / f"../../models/{model_name}.torchscript"
    input_batch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    if model_path.exists():
        return input_batch, model_path

    model = get_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        traced_model = torch.jit.trace(model, input_batch)

    torch.jit.save(traced_model, model_path)

    assert model_path.exists()
    return input_batch, model_path


def load_onnx(model_name):
    model_path = Path(__file__).parent / f"../../models/{model_name}.onnx"
    input_batch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    if model_path.exists():
        return input_batch, model_path

    model = get_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_batch, model_path, export_params=True, opset_version=11)

    assert model_path.exists()
    return input_batch, model_path


def benchmark_max(model_name):
    print("Benchmarking MAX...")
    print("Loading model...")
    input_batch, model_path = load_torchscript(model_name)

    session = engine.InferenceSession()
    model = session.load(model_path, input_specs=[engine.TorchInputSpec(shape=(1, 3, 224, 224), dtype=engine.DType.float32)])

    print("Warming up...")
    for _ in range(10):
        model.execute(x=input_batch)

    print("Benchmarking...")
    start = time.time()

    for _ in range(100):
        model.execute(x=input_batch)

    end = time.time()

    print(f"Time taken: {end - start:.2f} seconds")


def benchmark_torchscript(model_name):
    print("Benchmarking TorchScript...")
    print("Loading model...")
    input_batch, model_path = load_torchscript(model_name)

    model = torch.jit.load(model_path)

    print("Warming up...")
    for _ in range(10):
        model(input_batch)

    print("Benchmarking...")
    start = time.time()

    for _ in range(100):
        model(input_batch)

    end = time.time()

    print(f"Time taken: {end - start:.2f} seconds")


def benchmark_ort(model_name):
    print("Benchmarking ONNX Runtime...")
    print("Loading model...")
    input_batch, model_path = load_onnx(model_name)
    ort_session = ort.InferenceSession(str(model_path), prividers=["CPUExecutionProvider"])

    print("Warming up...")
    for _ in range(10):
        ort_session.run(None, {"input.1": input_batch.numpy()})

    print("Benchmarking...")
    start = time.time()

    for _ in range(100):
        input_batch = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        out = ort_session.run(None, {"input.1": input_batch.numpy()})

    end = time.time()

    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    model_name = "mobilenet_v3_large"
    benchmark_max(model_name)
    benchmark_torchscript(model_name)
    benchmark_ort(model_name)
