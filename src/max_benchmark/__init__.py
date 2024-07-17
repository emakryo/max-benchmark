import time

import numpy as np
import torch
import onnxruntime as ort
import pandas as pd
from max import engine

from max_benchmark import models


def benchmark_max(model_path, input):
    print("Benchmarking MAX...")
    session = engine.InferenceSession()
    model = session.load(model_path, input_specs=[engine.TorchInputSpec(shape=input.shape, dtype=engine.DType.float32)])
    input_name = model.input_metadata[0].name

    print("Warming up...")
    for _ in range(10):
        model.execute(**{input_name: input})

    print("Benchmarking...")
    elapsed = 0

    for _ in range(100):
        inputs = {input_name: np.random.random(input.shape).astype(np.float32)}

        start = time.time()
        model.execute(**inputs)
        end = time.time()

        elapsed += end - start

    print(f"Time taken: {elapsed:.2f} seconds")
    return elapsed


def benchmark_torchscript(model_path, input):
    print("Benchmarking TorchScript...")

    model = torch.jit.load(model_path)
    input = torch.tensor(input)

    print("Warming up...")
    for _ in range(10):
        model(input)

    print("Benchmarking...")
    elapsed = 0

    for _ in range(100):
        input = torch.tensor(np.random.random(input.shape).astype(np.float32))

        start = time.time()
        model(input)
        end = time.time()

        elapsed += end - start

    print(f"Time taken: {elapsed:.2f} seconds")
    return elapsed


def benchmark_ort(model_path, input):
    print("Benchmarking ONNX Runtime...")
    print("Loading model...")
    ort_session = ort.InferenceSession(model_path, prividers=["CPUExecutionProvider"])

    print("Warming up...")
    for _ in range(10):
        ort_session.run(None, {"input": input})

    print("Benchmarking...")

    elapsed = 0

    for _ in range(100):
        inputs = {"input": np.random.random(input.shape).astype(np.float32)}

        start = time.time()
        ort_session.run(None, inputs)
        end = time.time()

        elapsed += end - start

    print(f"Time taken: {elapsed:.2f} seconds")
    return elapsed


def main():
    model_names = [
        ("ultralytics", "yolov8n-seg"),
        # ("torchvision", "vit_l_16"),
        ("timm", "tf_efficientnet_lite0"),
        ("torchvision", "mobilenet_v2"),
        ("torchvision", "mobilenet_v3_large"),
        ("torchvision", "resnet50"),
    ]

    batch_sizes = [1]
    # batch_sizes = [1, 128]

    result = []

    for (source, model_name) in model_names:
        for batch_size in batch_sizes:
            print(f"Benchmarking {model_name} with batch size {batch_size}...")
            try:
                if source == "torchvision":
                    torch_model, input = models.torchvision.load_torchscript(model_name, batch_size)
                elif source == "timm":
                    torch_model, input = models.timm.load_torchscript(model_name, batch_size)
                elif source == "ultralytics":
                    torch_model, input = models.ultralytics.load_torchscript(model_name, batch_size)
            
                max_result = benchmark_max(torch_model, input)
                torch_result = benchmark_torchscript(torch_model, input)
            except Exception as e:
                raise e
                print("Exception occurred on torchscript model: ", model_name)
                print(e)
                max_result = float('nan')
                torch_result = float('nan')

            try:
                if source == "torchvision":
                    onnx_model, input = models.torchvision.load_onnx(model_name, batch_size)
                elif source == "timm":
                    onnx_model, input = models.timm.load_onnx(model_name, batch_size)
                elif source == "ultralytics":
                    onnx_model, input = models.ultralytics.load_onnx(model_name, batch_size)
            
                ort_result = benchmark_ort(onnx_model, input)
                onnx_max_result = benchmark_max(onnx_model, input)
            except Exception as e:
                raise e
                print("Exception occurred on onnx model: ", model_name)
                print(e)
                ort_result = float('nan')
                onnx_max_result = float('nan')

            result.append({
                "model": model_name,
                "batch_size": batch_size,
                "max_torch": max_result,
                "max_onnx": onnx_max_result,
                "torch": torch_result,
                "ort": ort_result,
            })

    pd.DataFrame(result).to_csv("benchmark.csv", index=False)

if __name__ == "__main__":
    main()
