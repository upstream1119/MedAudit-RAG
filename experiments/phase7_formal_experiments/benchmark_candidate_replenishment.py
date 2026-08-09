from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


NON_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+")
ALLOWED_DECISIONS = {"review_required", "insufficient_evidence", "boundary_refusal"}


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_question(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return NON_ALNUM_CJK_RE.sub("", normalized)


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(_normalize_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _character_ngrams(text: Any, n: int) -> set[str]:
    normalized = normalize_question(text)
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _jaccard(left: Any, right: Any, n: int) -> float:
    left_grams = _character_ngrams(left, n)
    right_grams = _character_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _anchor_map(anchors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id", ""))
        if not anchor_id or anchor_id in result:
            raise ValueError(f"锚点 ID 为空或重复: {anchor_id}")
        if anchor.get("verification_status") != "author_verified_anchor":
            raise ValueError(f"锚点未经作者核验: {anchor_id}")
        result[anchor_id] = anchor
    return result


def _parent_by_anchor(parents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for parent in parents:
        anchor_ids = parent.get("evidence_anchor_ids") or []
        if len(anchor_ids) != 1:
            continue
        anchor_id = str(anchor_ids[0])
        result.setdefault(anchor_id, parent)
    return result


def _validate_config(config: dict[str, Any]) -> None:
    specs = config.get("challenge_specs")
    if not isinstance(specs, list):
        raise ValueError("补充候选配置必须显式提供 challenge_specs")
    expected_count = int(config.get("expected_candidate_count", -1))
    if len(specs) != expected_count:
        raise ValueError(f"补充候选数量应为 {expected_count}，实际为 {len(specs)}")
    decisions = Counter(str(spec.get("provisional_expected_decision", "")) for spec in specs)
    if set(decisions) - ALLOWED_DECISIONS:
        raise ValueError("补充候选只能使用 review/insufficient/boundary 三类决策")
    expected_distribution = Counter(
        {key: int(value) for key, value in config["target_decision_distribution"].items()}
    )
    if decisions != expected_distribution:
        raise ValueError(f"补充候选决策分布不一致: {dict(decisions)}")
    if int(config.get("external_model_calls", -1)) != 0:
        raise ValueError("补充候选构建不允许外部模型调用")


def build_supplement_candidates(
    anchors: list[dict[str, Any]],
    parent_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build explicit evidence-boundary candidates without generating new facts."""
    _validate_config(config)
    anchors_by_id = _anchor_map(anchors)
    parents_by_anchor = _parent_by_anchor(parent_candidates)
    used_anchor_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for spec in config["challenge_specs"]:
        anchor_id = str(spec.get("anchor_id", ""))
        if anchor_id in used_anchor_ids:
            raise ValueError(f"补充候选重复使用锚点: {anchor_id}")
        used_anchor_ids.add(anchor_id)
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            raise ValueError(f"补充候选绑定未知锚点: {anchor_id}")
        parent = parents_by_anchor.get(anchor_id)
        if parent is None:
            raise ValueError(f"补充候选找不到父候选的独立性信息: {anchor_id}")
        question = _normalize_text(spec.get("question"))
        if not question:
            raise ValueError(f"补充候选问题为空: {anchor_id}")
        decision = str(spec["provisional_expected_decision"])
        candidate = {
            "candidate_id": _stable_id(
                "PMSQA-BV1S",
                anchor_id,
                question,
                config["config_version"],
            ),
            "question": question,
            "language": "zh-CN",
            "candidate_role": "evidence_boundary_challenge",
            "challenge_type": str(spec["challenge_type"]),
            "provisional_expected_decision": decision,
            "provisional_scenario_type": (
                "prescription-boundary" if decision == "boundary_refusal" else "evidence-scope"
            ),
            "provisional_risk_labels": sorted(set(spec["provisional_risk_labels"])),
            "current_kb_support": str(spec["current_kb_support"]),
            "missing_evidence_type": list(spec["missing_evidence_type"]),
            "policy_rule_ids": list(spec["policy_rule_ids"]),
            "source_id": anchor["source_id"],
            "source_title": anchor["source_title"],
            "source_filename": anchor["source_filename"],
            "source_sha256": anchor["source_sha256"],
            "page_number": int(anchor["page_number"]),
            "anchor_text_span": anchor["text_span"],
            "supported_claim_types": list(anchor["supported_claim_types"]),
            "evidence_scope": anchor["evidence_scope"],
            "age_scope": anchor["age_scope"],
            "applicability_conditions": anchor["applicability_conditions"],
            "scope_check": anchor["scope_check"],
            "evidence_anchor_ids": [anchor_id],
            "evidence_anchor_group_id": parent["evidence_anchor_group_id"],
            "provisional_fact_cluster_id": parent["provisional_fact_cluster_id"],
            "independence_unit_id": parent["independence_unit_id"],
            "candidate_status": config["candidate_status"],
            "candidate_generation_method": "explicit_evidence_boundary_spec",
            "annotation_status": "draft",
            "freeze_status": "draft",
            "dataset_version": config["dataset_version"],
            "schema_version": config["schema_version"],
            "protocol_version": config["protocol_version"],
            "kb_version": config["kb_version"],
            "generator_version": config["generator_version"],
            "config_version": config["config_version"],
        }
        candidates.append(candidate)
    normalized = [normalize_question(row["question"]) for row in candidates]
    if len(set(normalized)) != len(normalized):
        raise ValueError("补充候选内部存在完全重复问题")
    return sorted(candidates, key=lambda row: row["candidate_id"])


def audit_parent_overlap(
    candidates: list[dict[str, Any]],
    parent_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ngram_size = int(config["ngram_size"])
    threshold = float(config["jaccard_threshold"])
    audited: list[dict[str, Any]] = []
    for candidate in candidates:
        row = deepcopy(candidate)
        exact_parent = ""
        max_similarity = 0.0
        max_parent_id = ""
        for parent in parent_candidates:
            similarity = _jaccard(candidate["question"], parent.get("question"), ngram_size)
            if similarity > max_similarity:
                max_similarity = similarity
                max_parent_id = str(parent.get("candidate_id", ""))
            if normalize_question(candidate["question"]) == normalize_question(parent.get("question")):
                exact_parent = str(parent.get("candidate_id", ""))
                break
        reasons: list[str] = []
        decision = "keep"
        if exact_parent:
            reasons.append("exact_question_parent")
            decision = "reject"
        elif max_similarity >= threshold:
            reasons.append("near_duplicate_parent")
            decision = "needs_review"
        row.update(
            {
                "parent_overlap_decision": decision,
                "parent_overlap_reasons": reasons,
                "max_parent_question_similarity": round(max_similarity, 6),
                "max_parent_similar_candidate_id": exact_parent or max_parent_id,
            }
        )
        audited.append(row)
    return {
        "audited_candidates": audited,
        "decision_distribution": dict(
            sorted(Counter(row["parent_overlap_decision"] for row in audited).items())
        ),
        "usage": zero_usage(),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_sibling(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    spec.loader.exec_module(module)
    return module


def run_replenishment(
    *,
    anchor_path: str | Path,
    parent_candidate_path: str | Path,
    dev50_path: str | Path,
    dev50_registry_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    candidate_output_path: str | Path,
    overlap_output_path: str | Path,
    pass1_output_path: str | Path,
) -> dict[str, Any]:
    anchors = _load_jsonl(anchor_path)
    parents = _load_jsonl(parent_candidate_path)
    dev50_rows = _load_jsonl(dev50_path)
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)

    draft_candidates = build_supplement_candidates(anchors, parents, config)
    parent_audit = audit_parent_overlap(
        draft_candidates,
        parents,
        config["parent_overlap_config"],
    )
    unresolved_parent = [
        row
        for row in parent_audit["audited_candidates"]
        if row["parent_overlap_decision"] != "keep"
    ]
    if unresolved_parent:
        raise ValueError(
            f"补充候选与父候选存在 {len(unresolved_parent)} 条未解决重叠"
        )

    overlap_module = _load_sibling(
        "benchmark_overlap_audit_for_supplement",
        "benchmark_overlap_audit.py",
    )
    overlap_config = deepcopy(config["overlap_config"])
    dev50_registry = overlap_module.load_dev50_registry(dev50_registry_path)
    overlap_result = overlap_module.audit_candidate_overlap(
        parent_audit["audited_candidates"],
        dev50_rows,
        dev50_registry,
        overlap_config,
    )
    selected = overlap_module.select_deduplicated_candidates(
        overlap_result,
        overlap_config,
    )
    if len(selected) != int(config["expected_candidate_count"]):
        raise ValueError(
            "补充候选通过重叠审计后的数量不足，必须调整问题或补充新锚点"
        )
    expected_units = int(config["expected_candidate_count"])
    if len({row["independence_unit_id"] for row in selected}) != expected_units:
        raise ValueError("补充候选未保持一题一个不同独立性单元")

    annotation_module = _load_sibling(
        "benchmark_annotation_validator_for_supplement",
        "benchmark_annotation_validator.py",
    )
    annotation_config = deepcopy(config["annotation_config"])
    pass1_queue = annotation_module.build_pass1_queue(
        selected,
        anchors,
        manifest,
        annotation_config,
    )
    _write_jsonl(Path(candidate_output_path), selected)
    annotation_module.write_pass1_queue(pass1_queue, pass1_output_path)

    decision_distribution = dict(
        sorted(Counter(row["provisional_expected_decision"] for row in selected).items())
    )
    report = {
        "audit_version": config["audit_version"],
        "config_version": config["config_version"],
        "dataset_version": overlap_config["output_dataset_version"],
        "schema_version": config["schema_version"],
        "protocol_version": config["protocol_version"],
        "kb_version": config["kb_version"],
        "candidate_count": len(selected),
        "independence_unit_count": len(
            {row["independence_unit_id"] for row in selected}
        ),
        "decision_distribution": decision_distribution,
        "source_distribution": dict(
            sorted(Counter(row["source_id"] for row in selected).items())
        ),
        "parent_overlap_decision_distribution": parent_audit[
            "decision_distribution"
        ],
        "dev50_internal_review_queue_count": len(overlap_result["review_queue"]),
        "lineage": {
            "anchor_path": Path(anchor_path).as_posix(),
            "anchor_sha256": _sha256(anchor_path),
            "parent_candidate_path": Path(parent_candidate_path).as_posix(),
            "parent_candidate_sha256": _sha256(parent_candidate_path),
            "dev50_path": Path(dev50_path).as_posix(),
            "dev50_sha256": _sha256(dev50_path),
            "config_path": Path(config_path).as_posix(),
            "config_sha256": _sha256(config_path),
        },
        "audited_candidates": selected,
        "dev50_internal_review_queue": overlap_result["review_queue"],
        "pass1_queue_status": "pending_author_review",
        "usage": zero_usage(),
    }
    output = Path(overlap_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    benchmark_dir = root / "revision/benchmark/benchmark_v1"
    parser = argparse.ArgumentParser(
        description="Build and audit explicit Benchmark-v1 supplement candidates."
    )
    parser.add_argument("--anchors", default=benchmark_dir / "evidence_anchor_pool_v0_1.jsonl")
    parser.add_argument("--parents", default=benchmark_dir / "benchmark_candidates_v0_2_deduplicated.jsonl")
    parser.add_argument("--dev50", default=root / "revision/benchmark/dev50/dev50_v1_0_frozen.jsonl")
    parser.add_argument("--dev50-registry", default=root / "revision/benchmark/dev50/evidence_anchor_registry.md")
    parser.add_argument("--manifest", default=root / "data/guidelines/source_manifest.json")
    parser.add_argument("--config", default=Path(__file__).with_name("configs") / "benchmark_candidate_replenishment_v0_1.json")
    parser.add_argument("--candidates", default=benchmark_dir / "benchmark_supplement_candidates_v0_1.jsonl")
    parser.add_argument("--overlap-report", default=benchmark_dir / "benchmark_supplement_overlap_audit_v0_1.json")
    parser.add_argument("--pass1-queue", default=benchmark_dir / "benchmark_supplement_pass1_queue_v0_1.csv")
    args = parser.parse_args()
    report = run_replenishment(
        anchor_path=args.anchors,
        parent_candidate_path=args.parents,
        dev50_path=args.dev50,
        dev50_registry_path=args.dev50_registry,
        manifest_path=args.manifest,
        config_path=args.config,
        candidate_output_path=args.candidates,
        overlap_output_path=args.overlap_report,
        pass1_output_path=args.pass1_queue,
    )
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "decision_distribution": report["decision_distribution"],
                "review_queue_count": report["dev50_internal_review_queue_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
