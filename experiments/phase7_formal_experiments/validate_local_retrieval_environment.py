from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any


DEFAULT_TEXTS = [
    "儿童支原体肺炎使用阿奇霉素时应核对剂量与给药频次。",
    "儿童社区获得性肺炎治疗后四十八至七十二小时无改善时需要再次评估。",
]


def build_sparse_smoke_report(
    *,
    encoder: Any,
    texts: list[str],
    model_path: str,
    device: str,
    versions: dict[str, str],
    cuda_available: bool,
    gpu_name: str,
    offline_environment: dict[str, str],
) -> dict[str, Any]:
    outputs = encoder.encode(
        texts,
        batch_size=max(1, len(texts)),
        max_length=512,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    lexical_weights = outputs.get("lexical_weights") or []
    lexical_weight_counts = [len(weights) for weights in lexical_weights]
    non_empty_count = sum(count > 0 for count in lexical_weight_counts)
    offline_verified = all(
        offline_environment.get(name) == "1"
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    passed = (
        cuda_available
        and offline_verified
        and len(lexical_weights) == len(texts)
        and non_empty_count == len(texts)
    )

    return {
        "status": "passed" if passed else "failed",
        "model_path": model_path,
        "device": device,
        "versions": versions,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "offline_environment": offline_environment,
        "offline_loading_verified": offline_verified,
        "text_count": len(texts),
        "lexical_weight_counts": lexical_weight_counts,
        "non_empty_sparse_vector_count": non_empty_count,
        "lexical_weight_preview": [
            {str(token_id): float(weight) for token_id, weight in list(weights.items())[:5]}
            for weights in lexical_weights
        ],
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }


def run_real_smoke(model_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers
    from FlagEmbedding import BGEM3FlagModel

    if not model_path.is_dir():
        raise FileNotFoundError(f"Local BGE-M3 model not found: {model_path}")

    encoder = BGEM3FlagModel(
        str(model_path),
        use_fp16=True,
        devices=[device],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    report = build_sparse_smoke_report(
        encoder=encoder,
        texts=DEFAULT_TEXTS,
        model_path=str(model_path),
        device=device,
        versions={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "flag_embedding": version("FlagEmbedding"),
        },
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        offline_environment={
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["status"] != "passed":
        raise RuntimeError("Local sparse encoder smoke validation failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_real_smoke(args.model_path, args.output, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
