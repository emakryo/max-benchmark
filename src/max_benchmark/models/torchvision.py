from pathlib import Path

import torch 
from torchvision.models import get_model

import max_benchmark

MODEL_DIR = Path(max_benchmark.__file__).parent / "../../models/torchvision"


def load_torchscript(model_name, batch_size):
    model_path = MODEL_DIR / f"{model_name}.torchscript"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    input_batch = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32)

    if model_path.exists():
        return model_path, input_batch.numpy()

    model = get_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        traced_model = torch.jit.trace(model, input_batch)

    torch.jit.save(traced_model, model_path)

    assert model_path.exists()
    return model_path, input_batch.numpy()


def load_onnx(model_name, batch_size):
    model_path = MODEL_DIR / f"{model_name}.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    input_batch = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32)

    if model_path.exists():
        return model_path, input_batch.numpy()

    model = get_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_batch, model_path, input_names=["input"], export_params=True, opset_version=17)

    assert model_path.exists()
    return model_path, input_batch.numpy()
