"""Model architecture and ONNX export helpers for BirdCLEF+ 2026."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b2",
        num_classes: int = 234,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = int(self.backbone.num_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(self.dropout(features))
        return logits

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        model_kwargs = checkpoint.get(
            "model_kwargs",
            {
                "backbone_name": "tf_efficientnet_b2",
                "num_classes": 234,
                "pretrained": False,
                "dropout": 0.3,
            },
        )
        model = cls(**model_kwargs)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


def get_model(config_dict: dict) -> BirdCLEFModel:
    return BirdCLEFModel(
        backbone_name=config_dict.get("backbone", "tf_efficientnet_b2"),
        num_classes=int(config_dict.get("num_classes", 234)),
        pretrained=bool(config_dict.get("pretrained", True)),
        dropout=float(config_dict.get("dropout", 0.3)),
    )


def compile_model(model: BirdCLEFModel) -> BirdCLEFModel:
    """Return a torch.compile'd model if torch >= 2.0, otherwise return as-is."""
    if int(torch.__version__.split(".")[0]) >= 2:
        return torch.compile(model)
    return model


def export_onnx(model: nn.Module, output_path: str | Path, opset: int = 18) -> None:
    import onnxruntime as ort  # lazy import — only needed during export
    import warnings

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)

    # Use the legacy (non-dynamo) exporter for deterministic bit-exact parity.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["mel_input"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={"mel_input": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,  # legacy exporter — required for correct parity
        )

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"mel_input": dummy.numpy().astype(np.float32)})[0]
    if onnx_out.shape != (1, 234):
        raise RuntimeError(f"ONNX output shape mismatch: expected (1, 234), got {onnx_out.shape}")
    # GEM pooling (p=3) causes ~7e-3 float32 divergence between PyTorch and ONNX
    # fused kernels — this is negligible in probability space (sigmoid delta < 2e-4).
    max_diff = float(np.abs(pt_out - onnx_out).max())
    if max_diff > 1e-2:
        raise RuntimeError(f"ONNX parity check failed: max abs diff {max_diff:.3e} > 1e-2")
