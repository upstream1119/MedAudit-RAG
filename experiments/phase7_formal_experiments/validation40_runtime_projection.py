from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FIELDS = (
    "sample_id",
    "question",
    "dataset_version",
    "kb_version",
)

GOLD_ONLY_FIELDS = {
    "expected_decision",
    "required_evidence_type",
    "required_claims",
    "allowed_claims",
    "forbidden_claims",
    "risk_labels",
    "missing_evidence_type",
    "missing_information",
    "gold_evidence_status",
    "anchor_text_span",
    "page_number",
    "source_title",
    "source_filename",
}

OUTPUT_FILENAMES = {
    "runtime_rows": "validation40_runtime_projection_v0_1.jsonl",
    "selection_rows": "validation40_selection_manifest_v0_1.jsonl",
    "audit": "validation40_projection_audit_v0_1.json",
    "summary_markdown": "validation40_projection_summary_v0_1.md",
}


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at line {line_number}: {path}")
        rows.append(row)
    return rows


def _require_equal(row: dict[str, Any], field: str, expected: str) -> None:
    observed = _text(row.get(field))
    if observed != expected:
        sample = _text(row.get("candidate_id")) or "<unknown>"
        raise ValueError(
            f"{sample} {field} mismatch: expected={expected}, observed={observed}"
        )


def build_runtime_projection(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_source_sha256: str,
) -> dict[str, Any]:
    expected_hash = _text(config.get("expected_source_sha256")).lower()
    observed_hash = _text(observed_source_sha256).lower()
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(
            f"source hash mismatch: expected={expected_hash}, observed={observed_hash}"
        )

    expected_count = int(config.get("expected_count", 0))
    if len(rows) != expected_count:
        raise ValueError(
            f"record count mismatch: expected={expected_count}, observed={len(rows)}"
        )

    required_config_fields = (
        "projection_version",
        "expected_dataset_split",
        "expected_dataset_version",
        "expected_kb_version",
        "expected_freeze_version",
    )
    missing_config = [field for field in required_config_fields if not _text(config.get(field))]
    if missing_config:
        raise ValueError(f"config fields missing: {', '.join(missing_config)}")

    runtime_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        candidate_id = _text(row.get("candidate_id"))
        question = _text(row.get("question"))
        if not candidate_id or not question:
            raise ValueError("candidate_id and question are required")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)

        _require_equal(row, "dataset_split", config["expected_dataset_split"])
        _require_equal(row, "dataset_version", config["expected_dataset_version"])
        _require_equal(row, "kb_version", config["expected_kb_version"])
        _require_equal(row, "freeze_version", config["expected_freeze_version"])
        _require_equal(row, "freeze_status", "frozen")
        _require_equal(row, "split_status", "frozen")

        runtime_rows.append(
            {
                "sample_id": candidate_id,
                "question": question,
                "dataset_version": config["expected_dataset_version"],
                "kb_version": config["expected_kb_version"],
            }
        )
        selection_rows.append(
            {
                "selection_rank": rank,
                "sample_id": candidate_id,
                "selected_from_dataset_version": config["expected_dataset_version"],
                "kb_version": config["expected_kb_version"],
                "source_dataset_sha256": observed_hash,
                "projection_version": config["projection_version"],
                "selection_status": "frozen_validation_projection",
            }
        )

    leaked_fields = sorted(
        {
            field
            for row in runtime_rows
            for field in row
            if field in GOLD_ONLY_FIELDS or field not in RUNTIME_FIELDS
        }
    )
    if leaked_fields:
        raise ValueError(f"Gold field leakage detected: {', '.join(leaked_fields)}")

    audit = {
        "status": "validation40_runtime_projection_ready",
        "projection_version": config["projection_version"],
        "record_count": len(runtime_rows),
        "source_dataset_sha256": observed_hash,
        "source_dataset_version": config["expected_dataset_version"],
        "source_dataset_split": config["expected_dataset_split"],
        "freeze_version": config["expected_freeze_version"],
        "kb_version": config["expected_kb_version"],
        "runtime_fields": list(RUNTIME_FIELDS),
        "gold_only_fields": sorted(GOLD_ONLY_FIELDS),
        "gold_field_leakage_count": 0,
        "pilot_test_accessed": False,
        "clinically_validated": False,
        "usage": zero_usage(),
    }
    summary_markdown = "\n".join(
        [
            "# Validation40 runtime projection v0.1",
            "",
            f"- Status: `{audit['status']}`",
            f"- Records: `{audit['record_count']}`",
            f"- Source SHA-256: `{observed_hash}`",
            f"- Runtime fields: `{', '.join(RUNTIME_FIELDS)}`",
            "- Gold field leakage: `0`",
            "- Pilot Test80 accessed: `false`",
            "- External model calls: `0`",
            "- Clinically validated: `false`",
            "",
        ]
    )
    return {
        "runtime_rows": runtime_rows,
        "selection_rows": selection_rows,
        "audit": audit,
        "summary_markdown": summary_markdown,
    }


def _projection_payloads(result: dict[str, Any]) -> dict[str, bytes]:
    return {
        OUTPUT_FILENAMES["runtime_rows"]: _canonical_jsonl_bytes(
            result["runtime_rows"]
        ),
        OUTPUT_FILENAMES["selection_rows"]: _canonical_jsonl_bytes(
            result["selection_rows"]
        ),
        OUTPUT_FILENAMES["audit"]: _canonical_json_bytes(result["audit"]),
        OUTPUT_FILENAMES["summary_markdown"]: result["summary_markdown"].encode(
            "utf-8"
        ),
    }


def write_projection_outputs(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    output_path = Path(output_dir)
    payloads = _projection_payloads(result)

    for filename, content in payloads.items():
        target = output_path / filename
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"output conflict: refusing to overwrite {target}")

    output_path.mkdir(parents=True, exist_ok=True)
    for filename, content in payloads.items():
        target = output_path / filename
        if target.exists():
            continue
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=output_path, prefix=f".{filename}."
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(target)

    return {
        filename: _sha256_bytes(content) for filename, content in payloads.items()
    }


def run(config_path: str | Path, *, repo_root: str | Path = ROOT) -> dict[str, Any]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8-sig"))
    root = Path(repo_root)
    source_path = root / _text(config.get("source_path"))
    output_dir = root / _text(config.get("output_dir"))
    observed_hash = compute_sha256(source_path)
    rows = _read_jsonl(source_path)
    result = build_runtime_projection(
        rows,
        config,
        observed_source_sha256=observed_hash,
    )
    output_hashes = write_projection_outputs(result, output_dir)
    return {
        **result["audit"],
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "output_sha256": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gold-isolated runtime projection for frozen Validation40."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    summary = run(args.config, repo_root=args.repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
