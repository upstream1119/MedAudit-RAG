from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_anchor_review as anchor_review


ANCHOR_FIELDS = (
    "age_scope",
    "anchor_id",
    "applicability_conditions",
    "author_review_reason",
    "author_reviewed_at",
    "candidate_config_version",
    "candidate_id",
    "chunker_version",
    "evidence_scope",
    "page_number",
    "parser_version",
    "review_config_version",
    "reviewer_id",
    "scope_check",
    "source_filename",
    "source_id",
    "source_sha256",
    "source_title",
    "supported_claim_types",
    "text_span",
    "verification_status",
)

DECISION_FIELDS = (
    "candidate_id",
    "source_id",
    "page_number",
    "author_review_outcome",
    "promotion_decision",
    "reason",
    "anchor_id",
)

REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "promotion_version",
    "dataset_version",
    "kb_version",
    "expected_review_count",
    "expected_accepted_count",
    "expected_rejected_count",
    "expected_existing_anchor_count",
    "expected_expansion_anchor_count",
    "expected_merged_anchor_count",
    "expected_parent_queue_sha256",
    "expected_author_review_sha256",
    "expected_author_audit_sha256",
    "expected_coverage_sha256",
    "expected_existing_pool_sha256",
    "expected_dev50_registry_sha256",
    "min_verified_text_chars",
    "required_scope_check",
}

HASH_BINDINGS = {
    "parent_queue_sha256": "expected_parent_queue_sha256",
    "author_review_sha256": "expected_author_review_sha256",
    "author_audit_sha256": "expected_author_audit_sha256",
    "coverage_sha256": "expected_coverage_sha256",
    "existing_pool_sha256": "expected_existing_pool_sha256",
    "dev50_registry_sha256": "expected_dev50_registry_sha256",
}

IMMUTABLE_PARENT_FIELDS = (
    "candidate_id",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "page_number",
    "block_type",
    "granularity",
    "candidate_text",
    "context_candidate_id",
    "context_text",
    "matched_topics",
    "matched_terms",
    "review_status",
    "dev50_overlap_anchor_ids",
    "selection_rank_within_source",
    "parser_version",
    "chunker_version",
    "candidate_config_version",
    "review_config_version",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def load_promotion_config(path: str | Path) -> dict[str, Any]:
    config = _load_json(path)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"promotion config missing fields: {', '.join(missing)}")
    count_fields = [field for field in REQUIRED_CONFIG_FIELDS if "count" in field]
    if any(int(config[field]) < 0 for field in count_fields):
        raise ValueError("promotion config counts must be non-negative")
    if int(config["min_verified_text_chars"]) <= 0:
        raise ValueError("min_verified_text_chars must be positive")
    return config


def _normalized_scalar(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def _normalized_text(value: Any) -> str:
    return anchor_review._normalize_text(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    return text.encode("utf-8")


def _decision_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=DECISION_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _validate_hashes(config: dict[str, Any], artifact_hashes: dict[str, str]) -> None:
    for actual_key, expected_key in HASH_BINDINGS.items():
        actual = str(artifact_hashes.get(actual_key, "")).lower()
        expected = str(config[expected_key]).lower()
        if actual != expected:
            raise ValueError(
                f"hash drift: {actual_key} expected {expected}, got {actual or 'missing'}"
            )


def _unique_map(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise ValueError(f"{label} contains an empty {key}")
        if value in result:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        result[value] = row
    return result


def _validate_existing_pool(existing_anchors: list[dict[str, Any]]) -> None:
    anchor_ids: set[str] = set()
    source_pages: set[tuple[str, int]] = set()
    expected_fields = set(ANCHOR_FIELDS)
    for anchor in existing_anchors:
        if set(anchor) != expected_fields:
            raise ValueError("existing anchor schema drift")
        anchor_id = str(anchor["anchor_id"])
        pair = (str(anchor["source_id"]), int(anchor["page_number"]))
        if anchor_id in anchor_ids:
            raise ValueError(f"duplicate existing anchor_id: {anchor_id}")
        if pair in source_pages:
            raise ValueError(f"duplicate existing source/page: {pair}")
        anchor_ids.add(anchor_id)
        source_pages.add(pair)


def _validate_parent_link(author: dict[str, Any], parent: dict[str, Any]) -> None:
    failed: list[str] = []
    for field in IMMUTABLE_PARENT_FIELDS:
        left = _normalized_scalar(author.get(field))
        right = _normalized_scalar(parent.get(field))
        if left != right:
            failed.append(field)
    if failed:
        prefix = "page provenance mismatch" if "page_number" in failed else "provenance mismatch"
        raise ValueError(f"{prefix}: {author.get('candidate_id')} / {', '.join(failed)}")


def _validate_traceable_text(author: dict[str, Any], config: dict[str, Any]) -> str:
    text = _normalized_text(author.get("verified_text_span"))
    if len(text) < int(config["min_verified_text_chars"]):
        raise ValueError(f"trace validation failed: {author.get('candidate_id')} text too short")
    if "�" in text:
        raise ValueError(f"trace validation failed: {author.get('candidate_id')} garbled text")
    candidate_text = _normalized_text(author.get("candidate_text"))
    context_text = _normalized_text(author.get("context_text"))
    if text not in candidate_text and text not in context_text:
        raise ValueError(f"trace validation failed: {author.get('candidate_id')} text is not traceable")
    return text


def _validate_author_audit(
    audit: dict[str, Any],
    config: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    expected_counts = {
        "author_reviewed_count": int(config["expected_review_count"]),
    }
    for field, expected in expected_counts.items():
        if int(audit.get(field, -1)) != expected:
            raise ValueError(f"author audit count drift: {field}")
    outcomes = audit.get("outcome_counts") or {}
    if int(outcomes.get("accepted", -1)) != int(config["expected_accepted_count"]):
        raise ValueError("author audit accepted count drift")
    if int(outcomes.get("rejected", -1)) != int(config["expected_rejected_count"]):
        raise ValueError("author audit rejected count drift")
    if audit.get("status") != "author_review_complete":
        raise ValueError("author audit status is not complete")
    if audit.get("anchor_promotion_performed") is not False:
        raise ValueError("author audit unexpectedly reports prior promotion")
    if str(audit.get("parent_queue_sha256", "")).lower() != str(
        config["expected_parent_queue_sha256"]
    ).lower():
        raise ValueError("author audit parent hash drift")
    if str(audit.get("author_review_csv_sha256", "")).lower() != str(
        artifact_hashes["author_review_sha256"]
    ).lower():
        raise ValueError("author audit review hash drift")


def build_promotion_result(
    *,
    author_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    existing_anchors: list[dict[str, Any]],
    dev50_pairs: dict[tuple[str, int], list[str]],
    page_counts: dict[str, int],
    config: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """独立校验作者结论，并构建不覆盖旧池的 v0.2 promotion 结果。"""
    _validate_hashes(config, artifact_hashes)
    _validate_existing_pool(existing_anchors)

    if len(author_rows) != int(config["expected_review_count"]):
        raise ValueError("author review count drift")
    if len(parent_rows) != int(config["expected_review_count"]):
        raise ValueError("parent queue count drift")
    if len(existing_anchors) != int(config["expected_existing_anchor_count"]):
        raise ValueError("existing anchor count drift")

    author_map = _unique_map(author_rows, "candidate_id", "author review")
    parent_map = _unique_map(parent_rows, "candidate_id", "parent queue")
    coverage_map = _unique_map(coverage_rows, "source_id", "coverage")
    if set(author_map) != set(parent_map):
        raise ValueError("author review and parent queue candidate coverage mismatch")

    outcome_counts = Counter(str(row.get("author_review_outcome", "")) for row in author_rows)
    if outcome_counts != Counter(
        {
            "accepted": int(config["expected_accepted_count"]),
            "rejected": int(config["expected_rejected_count"]),
        }
    ):
        raise ValueError(f"author outcome count drift: {dict(outcome_counts)}")

    existing_ids = {str(row["anchor_id"]) for row in existing_anchors}
    existing_pairs = {
        (str(row["source_id"]), int(row["page_number"])) for row in existing_anchors
    }
    new_ids: set[str] = set()
    new_pairs: set[tuple[str, int]] = set()
    new_anchors: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for candidate_id in sorted(author_map):
        author = author_map[candidate_id]
        parent = parent_map[candidate_id]
        _validate_parent_link(author, parent)
        source_id = str(author.get("source_id", ""))
        source = coverage_map.get(source_id)
        try:
            page_number = int(author.get("page_number", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"page validation failed: {candidate_id}") from error

        decision = {
            "candidate_id": candidate_id,
            "source_id": source_id,
            "page_number": page_number,
            "author_review_outcome": str(author.get("author_review_outcome", "")),
            "promotion_decision": "",
            "reason": "",
            "anchor_id": "",
        }
        if author.get("author_review_outcome") == "rejected":
            if not _normalized_text(author.get("author_review_reason")):
                raise ValueError(f"rejected row lacks reason: {candidate_id}")
            decision["promotion_decision"] = "excluded_author_rejected"
            decision["reason"] = str(author["author_review_reason"])
            decisions.append(decision)
            continue

        if source is None:
            raise ValueError(f"provenance mismatch: unknown source {source_id}")
        if not source.get("included_in_kb"):
            raise ValueError(f"provenance mismatch: source excluded from KB {source_id}")
        provenance_checks = {
            "source_title": author.get("source_title") == source.get("title"),
            "source_filename": author.get("source_filename") == source.get("filename"),
            "source_sha256": (
                author.get("source_sha256")
                == source.get("actual_sha256")
                == source.get("recorded_sha256")
            ),
        }
        failed_provenance = [field for field, passed in provenance_checks.items() if not passed]
        if failed_provenance:
            raise ValueError(
                f"provenance mismatch: {candidate_id} / {', '.join(failed_provenance)}"
            )
        if page_number <= 0 or page_number > int(page_counts.get(source_id, 0)):
            raise ValueError(f"page validation failed: {candidate_id} / {page_number}")
        pair = (source_id, page_number)
        if pair in dev50_pairs:
            raise ValueError(f"Dev50 overlap: {candidate_id} / {pair}")
        if pair in existing_pairs or pair in new_pairs:
            raise ValueError(f"duplicate source/page: {candidate_id} / {pair}")
        if author.get("scope_check") != config["required_scope_check"]:
            raise ValueError(f"scope validation failed: {candidate_id}")

        required_author_fields = (
            "reviewer_id",
            "author_reviewed_at",
            "author_review_reason",
            "evidence_scope",
            "age_scope",
            "applicability_conditions",
        )
        missing = [field for field in required_author_fields if not _normalized_text(author.get(field))]
        if missing:
            raise ValueError(f"author metadata incomplete: {candidate_id} / {', '.join(missing)}")
        if not anchor_review._author_metadata_is_readable(author):
            raise ValueError(f"author metadata garbled: {candidate_id}")

        text_span = _validate_traceable_text(author, config)
        try:
            supported_claim_types = anchor_review._parse_list(
                author.get("supported_claim_types")
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"supported claim types invalid: {candidate_id}") from error
        if not supported_claim_types or not all(
            isinstance(value, str) and value.strip() for value in supported_claim_types
        ):
            raise ValueError(f"supported claim types invalid: {candidate_id}")

        anchor_id = anchor_review._anchor_id(source_id, page_number, text_span)
        if anchor_id in existing_ids or anchor_id in new_ids:
            raise ValueError(f"duplicate anchor_id: {candidate_id} / {anchor_id}")
        anchor = {
            "age_scope": str(author["age_scope"]),
            "anchor_id": anchor_id,
            "applicability_conditions": str(author["applicability_conditions"]),
            "author_review_reason": str(author["author_review_reason"]),
            "author_reviewed_at": str(author["author_reviewed_at"]),
            "candidate_config_version": str(author.get("candidate_config_version", "")),
            "candidate_id": candidate_id,
            "chunker_version": str(author.get("chunker_version", "")),
            "evidence_scope": str(author["evidence_scope"]),
            "page_number": page_number,
            "parser_version": str(author.get("parser_version", "")),
            "review_config_version": str(author.get("review_config_version", "")),
            "reviewer_id": str(author["reviewer_id"]),
            "scope_check": str(author["scope_check"]),
            "source_filename": str(source["filename"]),
            "source_id": source_id,
            "source_sha256": str(source["actual_sha256"]),
            "source_title": str(source["title"]),
            "supported_claim_types": supported_claim_types,
            "text_span": text_span,
            "verification_status": "author_verified_anchor",
        }
        if set(anchor) != set(ANCHOR_FIELDS):
            raise AssertionError("new anchor schema mismatch")
        new_anchors.append(anchor)
        new_ids.add(anchor_id)
        new_pairs.add(pair)
        decision["promotion_decision"] = "promoted_author_verified"
        decision["reason"] = "passed independent hash, provenance, page, text, scope and leakage gates"
        decision["anchor_id"] = anchor_id
        decisions.append(decision)

    new_anchors.sort(key=lambda row: row["anchor_id"])
    decisions.sort(key=lambda row: row["candidate_id"])
    merged_anchors = sorted(
        deepcopy(existing_anchors) + deepcopy(new_anchors),
        key=lambda row: row["anchor_id"],
    )
    if len(new_anchors) != int(config["expected_expansion_anchor_count"]):
        raise ValueError("expansion anchor count drift")
    if len(merged_anchors) != int(config["expected_merged_anchor_count"]):
        raise ValueError("merged anchor count drift")

    new_bytes = _jsonl_bytes(new_anchors)
    merged_bytes = _jsonl_bytes(merged_anchors)
    decision_bytes = _decision_csv_bytes(decisions)
    audit = {
        "status": "promotion_validated",
        "config_version": config["config_version"],
        "promotion_version": config["promotion_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "reviewed_count": len(author_rows),
        "accepted_count": outcome_counts["accepted"],
        "rejected_count": outcome_counts["rejected"],
        "existing_anchor_count": len(existing_anchors),
        "expansion_anchor_count": len(new_anchors),
        "merged_anchor_count": len(merged_anchors),
        "anchor_promotion_performed": True,
        "input_sha256": dict(sorted(artifact_hashes.items())),
        "output_sha256": {
            "evidence_anchor_expansion_v0_2": _sha256_bytes(new_bytes),
            "evidence_anchor_pool_v0_2": _sha256_bytes(merged_bytes),
            "promotion_decisions_v0_2": _sha256_bytes(decision_bytes),
        },
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "medical_boundary": (
            "author-verified evidence anchors are guideline-grounded research artifacts; "
            "they are not expert-validated gold evidence or clinical validation."
        ),
    }
    return {
        "new_anchors": new_anchors,
        "merged_anchors": merged_anchors,
        "decisions": decisions,
        "audit": audit,
    }


def _summary_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Benchmark-v1 Anchor Expansion Promotion Summary v0.2",
            "",
            f"- 状态：`{audit['status']}`",
            f"- 作者核验记录：{audit['reviewed_count']}",
            f"- 独立门禁后升级：{audit['expansion_anchor_count']}",
            f"- 作者拒绝并排除：{audit['rejected_count']}",
            f"- 既有锚点：{audit['existing_anchor_count']}",
            f"- 合并后锚点：{audit['merged_anchor_count']}",
            "- 外部 API 调用：0",
            "- input/output tokens：0/0",
            "- estimated cost：0",
            "- 医学边界：本次结果为 author-verified、guideline-grounded 研究锚点，不等于独立专家验证、临床验证或最终 gold evidence。",
            "",
        ]
    )


def write_promotion_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payloads = {
        "evidence_anchor_expansion_v0_2.jsonl": _jsonl_bytes(result["new_anchors"]),
        "evidence_anchor_pool_v0_2.jsonl": _jsonl_bytes(result["merged_anchors"]),
        "anchor_expansion_promotion_decisions_v0_2.csv": _decision_csv_bytes(
            result["decisions"]
        ),
        "anchor_expansion_promotion_audit_v0_2.json": (
            json.dumps(result["audit"], ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"),
        "anchor_expansion_promotion_summary_v0_2.md": _summary_markdown(
            result["audit"]
        ).encode("utf-8"),
    }
    written: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="anchor-promotion-", dir=output_path) as temp_dir:
        temp_path = Path(temp_dir)
        for filename, payload in payloads.items():
            (temp_path / filename).write_bytes(payload)
        for filename in payloads:
            target = output_path / filename
            os.replace(temp_path / filename, target)
            written[filename] = target
    return written


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_promotion_config(args.config)
    paths = {
        "parent_queue_sha256": args.parent_queue,
        "author_review_sha256": args.author_review,
        "author_audit_sha256": args.author_audit,
        "coverage_sha256": args.coverage,
        "existing_pool_sha256": args.existing_pool,
        "dev50_registry_sha256": args.dev50_registry,
    }
    artifact_hashes = {
        key: anchor_review._compute_sha256(path) for key, path in paths.items()
    }
    author_rows = _load_csv(args.author_review)
    parent_rows = _load_csv(args.parent_queue)
    coverage_rows = _load_jsonl(args.coverage)
    existing_anchors = _load_jsonl(args.existing_pool)
    author_audit = _load_json(args.author_audit)
    _validate_hashes(config, artifact_hashes)
    _validate_author_audit(author_audit, config, artifact_hashes)
    result = build_promotion_result(
        author_rows=author_rows,
        parent_rows=parent_rows,
        coverage_rows=coverage_rows,
        existing_anchors=existing_anchors,
        dev50_pairs=anchor_review.load_dev50_anchor_pairs(args.dev50_registry),
        page_counts=anchor_review._build_page_counts(coverage_rows, args.formal_dir),
        config=config,
        artifact_hashes=artifact_hashes,
    )
    if args.mode == "promote":
        written = write_promotion_outputs(result, args.output_dir)
        result["written_files"] = {key: str(value) for key, value in written.items()}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently promote author-reviewed expansion anchors."
    )
    parser.add_argument("--mode", choices=("validate", "promote"), default="validate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/phase7_formal_experiments/configs/"
            "benchmark_anchor_expansion_promotion_v0_2.json"
        ),
    )
    parser.add_argument(
        "--parent-queue",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/anchor_expansion_review_queue_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--author-review",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/anchor_expansion_author_review_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--author-audit",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/anchor_expansion_author_review_audit_v0_2.json"
        ),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/source_coverage_matrix_v0_1.jsonl"
        ),
    )
    parser.add_argument(
        "--existing-pool",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/evidence_anchor_pool_v0_1.jsonl"
        ),
    )
    parser.add_argument(
        "--dev50-registry",
        type=Path,
        default=Path("revision/benchmark/dev50/evidence_anchor_registry.md"),
    )
    parser.add_argument("--formal-dir", type=Path, default=Path("data/guidelines"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["audit"]["status"],
                "accepted": result["audit"]["accepted_count"],
                "rejected": result["audit"]["rejected_count"],
                "merged": result["audit"]["merged_anchor_count"],
                "mode": args.mode,
                "written_files": result.get("written_files", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
