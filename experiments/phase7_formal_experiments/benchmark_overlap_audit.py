from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


NON_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
DEV50_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(SRC-\d+)\s*\|\s*(\d+)\s*\|"
)
REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "audit_version",
    "input_dataset_version",
    "output_dataset_version",
    "schema_version",
    "protocol_version",
    "kb_version",
    "ngram_size",
    "jaccard_threshold",
    "normalization_rules",
    "grouping_fields",
    "expected_candidate_count",
    "expected_dev50_count",
    "expected_independence_unit_count",
    "fail_on_unresolved_review",
    "audited_candidate_status",
    "external_model_calls",
}
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "question",
    "source_id",
    "page_number",
    "evidence_anchor_ids",
    "provisional_fact_cluster_id",
    "evidence_anchor_group_id",
    "candidate_status",
    "dataset_version",
    "schema_version",
    "protocol_version",
    "kb_version",
}
REVIEW_QUEUE_FIELDS = [
    "review_id",
    "candidate_id",
    "comparison_scope",
    "compared_id",
    "similarity",
    "candidate_question",
    "compared_question",
    "manual_decision",
    "reviewer_id",
    "reviewed_at",
    "review_reason",
]
OUTPUT_AUDIT = "overlap_audit_v0_1.json"
OUTPUT_REVIEW_QUEUE = "overlap_review_queue_v0_1.csv"
OUTPUT_CANDIDATES = "benchmark_candidates_v0_2_deduplicated.jsonl"
REVISION_REAUDIT_VERSION = "benchmark-revision-overlap-reaudit-v0.1"


def normalize_question(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return NON_ALNUM_CJK_RE.sub("", normalized)


def _compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_json(path)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"重叠审计配置缺少字段: {', '.join(missing)}")
    if int(config["ngram_size"]) <= 0:
        raise ValueError("ngram_size 必须大于 0")
    threshold = float(config["jaccard_threshold"])
    if not 0 < threshold <= 1:
        raise ValueError("jaccard_threshold 必须位于 (0, 1]")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("B2.2 禁止调用外部模型")
    return config


def load_dev50_registry(
    path: str | Path,
) -> dict[tuple[str, int], list[str]]:
    registry: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = DEV50_ROW_RE.match(line)
        if not match:
            continue
        anchor_id, source_id, page_number = match.groups()
        registry[(source_id, int(page_number))].append(anchor_id.strip())
    return {
        key: sorted(set(anchor_ids))
        for key, anchor_ids in sorted(registry.items())
    }


def _validate_inputs(
    candidates: list[dict[str, Any]],
    dev50_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if len(candidates) != int(config["expected_candidate_count"]):
        raise ValueError(
            "候选数量与配置不一致: "
            f"{len(candidates)} != {config['expected_candidate_count']}"
        )
    if len(dev50_rows) != int(config["expected_dev50_count"]):
        raise ValueError(
            "Dev50 数量与配置不一致: "
            f"{len(dev50_rows)} != {config['expected_dev50_count']}"
        )
    candidate_ids: set[str] = set()
    for candidate in candidates:
        missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(candidate))
        if missing:
            raise ValueError(
                f"{candidate.get('candidate_id', '<unknown>')} 缺少字段: "
                f"{', '.join(missing)}"
            )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"候选 ID 重复: {candidate_id}")
        candidate_ids.add(candidate_id)
        if candidate["candidate_status"] != "draft_candidate_unverified":
            raise ValueError(f"候选状态不允许进入 B2.2: {candidate_id}")
        if candidate["dataset_version"] != config["input_dataset_version"]:
            raise ValueError(f"候选 dataset_version 不一致: {candidate_id}")


def character_ngrams(text: Any, n: int = 3) -> set[str]:
    normalized = normalize_question(text)
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {
        normalized[index : index + n]
        for index in range(len(normalized) - n + 1)
    }


def jaccard_similarity(left: Any, right: Any, n: int = 3) -> float:
    left_grams = character_ngrams(left, n)
    right_grams = character_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union)


def _effective_pass1_question(row: dict[str, Any]) -> str:
    return str(row.get("pass1_final_question") or row.get("question") or "")


def audit_revised_question_overlap(
    revised_row: dict[str, Any],
    pass1_rows: list[dict[str, Any]],
    dev50_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-audit only the changed question while preserving the parent B2.2 audit."""
    candidate_id = str(revised_row.get("candidate_id", ""))
    original_question = str(revised_row.get("question", ""))
    revised_question = str(revised_row.get("pass1_final_question", ""))
    independence_unit_id = str(revised_row.get("independence_unit_id", ""))
    if not candidate_id or not revised_question or not independence_unit_id:
        raise ValueError("修订问题复审缺少 candidate_id、问题或独立性单元")
    if revised_row.get("pass1_outcome") != "revise":
        raise ValueError(f"{candidate_id} 不是 revise 记录")
    if normalize_question(original_question) == normalize_question(revised_question):
        raise ValueError(f"{candidate_id} 的修订问题未发生实质变化")

    ngram_size = int(config["ngram_size"])
    threshold = float(config["jaccard_threshold"])
    dev50_matches = [
        (
            jaccard_similarity(revised_question, row.get("question"), ngram_size),
            row,
        )
        for row in sorted(dev50_rows, key=lambda item: str(item.get("sample_id", "")))
    ]
    max_dev50_similarity, max_dev50_row = max(
        dev50_matches,
        key=lambda item: item[0],
        default=(0.0, {}),
    )
    exact_dev50 = any(
        normalize_question(revised_question) == normalize_question(row.get("question"))
        for row in dev50_rows
    )

    compared_rows = [
        row
        for row in pass1_rows
        if str(row.get("candidate_id", "")) != candidate_id
        and str(row.get("independence_unit_id", "")) != independence_unit_id
    ]
    internal_matches = [
        (
            jaccard_similarity(
                revised_question,
                _effective_pass1_question(row),
                ngram_size,
            ),
            row,
        )
        for row in sorted(compared_rows, key=lambda item: str(item.get("candidate_id", "")))
    ]
    max_internal_similarity, max_internal_row = max(
        internal_matches,
        key=lambda item: item[0],
        default=(0.0, {}),
    )
    exact_internal = any(
        normalize_question(revised_question)
        == normalize_question(_effective_pass1_question(row))
        for row in compared_rows
    )

    reasons: list[str] = []
    if exact_dev50:
        reasons.append("exact_question_dev50")
    if exact_internal:
        reasons.append("exact_question_internal")
    if not exact_dev50 and max_dev50_similarity >= threshold:
        reasons.append("near_duplicate_dev50")
    if not exact_internal and max_internal_similarity >= threshold:
        reasons.append("near_duplicate_internal")
    rejected = exact_dev50 or exact_internal
    needs_review = not rejected and bool(reasons)
    decision = "reject" if rejected else "needs_review" if needs_review else "clear"

    return {
        "reaudit_version": REVISION_REAUDIT_VERSION,
        "candidate_id": candidate_id,
        "original_question": original_question,
        "revised_question": revised_question,
        "independence_unit_id": independence_unit_id,
        "dev50_overlap_status": (
            "rejected"
            if exact_dev50
            else "needs_review"
            if max_dev50_similarity >= threshold
            else "clear"
        ),
        "max_dev50_question_similarity": round(max_dev50_similarity, 6),
        "max_dev50_similar_sample_id": str(max_dev50_row.get("sample_id", "")),
        "internal_overlap_status": (
            "rejected"
            if exact_internal
            else "needs_review"
            if max_internal_similarity >= threshold
            else "clear"
        ),
        "max_internal_question_similarity": round(max_internal_similarity, 6),
        "max_internal_similar_candidate_id": str(
            max_internal_row.get("candidate_id", "")
        ),
        "compared_candidate_count": len(compared_rows),
        "reaudit_decision": decision,
        "reaudit_reasons": reasons,
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
    }


def _independence_unit_id(fact_cluster_id: Any, anchor_group_id: Any) -> str:
    payload = f"{fact_cluster_id}|{anchor_group_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"IU-BV1-{digest}"


def audit_candidate_overlap(
    candidates: list[dict[str, Any]],
    dev50_rows: list[dict[str, Any]],
    dev50_registry: dict[tuple[str, int], list[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ngram_size = int(config["ngram_size"])
    threshold = float(config["jaccard_threshold"])
    dev50_questions = {
        normalize_question(row.get("question")) for row in dev50_rows
    }
    dev50_anchor_ids = {
        anchor_id
        for row in dev50_rows
        for anchor_id in row.get("evidence_anchor_ids", [])
    }
    dev50_fact_clusters = {
        row.get("fact_cluster_id")
        for row in dev50_rows
        if row.get("fact_cluster_id")
    }
    candidate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidate in candidates:
        group_key = (
            str(candidate.get("provisional_fact_cluster_id", "")),
            str(candidate.get("evidence_anchor_group_id", "")),
        )
        candidate_groups[group_key].append(str(candidate["candidate_id"]))
    candidate_groups = {
        group_key: sorted(candidate_ids)
        for group_key, candidate_ids in candidate_groups.items()
    }
    audited_candidates: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: str(item.get("candidate_id", "")),
    ):
        audited = deepcopy(candidate)
        reasons: list[str] = []
        if normalize_question(candidate.get("question")) in dev50_questions:
            reasons.append("exact_question")
        if set(candidate.get("evidence_anchor_ids", [])).intersection(
            dev50_anchor_ids
        ):
            reasons.append("evidence_anchor_id")
        if candidate.get("provisional_fact_cluster_id") in dev50_fact_clusters:
            reasons.append("fact_cluster_id")
        source_page = (
            candidate.get("source_id"),
            int(candidate.get("page_number", 0)),
        )
        overlap_anchor_ids = sorted(dev50_registry.get(source_page, []))
        if overlap_anchor_ids:
            reasons.append("source_page_anchor")
        dev50_matches = [
            (
                jaccard_similarity(
                    candidate.get("question"),
                    row.get("question"),
                    ngram_size,
                ),
                row,
            )
            for row in sorted(
                dev50_rows,
                key=lambda item: str(item.get("sample_id", "")),
            )
        ]
        max_similarity, most_similar_dev50 = max(
            dev50_matches,
            key=lambda item: item[0],
            default=(0.0, {}),
        )
        rejected = bool(reasons)
        needs_review = not rejected and max_similarity >= threshold
        if needs_review:
            reasons.append("near_duplicate_dev50")
            review_queue.append(
                {
                    "review_id": (
                        f"OVR-{candidate['candidate_id']}-"
                        f"{most_similar_dev50.get('sample_id', '')}"
                    ),
                    "candidate_id": candidate["candidate_id"],
                    "comparison_scope": "dev50",
                    "compared_id": most_similar_dev50.get("sample_id", ""),
                    "similarity": round(max_similarity, 6),
                    "candidate_question": candidate.get("question", ""),
                    "compared_question": most_similar_dev50.get("question", ""),
                    "manual_decision": "",
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "review_reason": "",
                }
            )
        decision = (
            "reject"
            if rejected
            else "needs_review"
            if needs_review
            else "keep"
        )
        group_key = (
            str(candidate.get("provisional_fact_cluster_id", "")),
            str(candidate.get("evidence_anchor_group_id", "")),
        )
        same_group_ids = candidate_groups[group_key]
        audited.update(
            {
                "dev50_overlap_status": (
                    "rejected"
                    if rejected
                    else "needs_review"
                    if needs_review
                    else "clear"
                ),
                "dev50_overlap_anchor_ids": overlap_anchor_ids,
                "max_dev50_question_similarity": round(max_similarity, 6),
                "max_dev50_similar_sample_id": most_similar_dev50.get(
                    "sample_id", ""
                ),
                "internal_overlap_status": (
                    "group_linked" if len(same_group_ids) > 1 else "clear"
                ),
                "independence_unit_id": _independence_unit_id(*group_key),
                "same_group_candidate_ids": same_group_ids,
                "max_internal_question_similarity": 0.0,
                "max_internal_similar_candidate_id": "",
                "overlap_decision": decision,
                "overlap_reasons": reasons,
            }
        )
        audited_candidates.append(audited)

    normalized_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audited in audited_candidates:
        normalized_groups[normalize_question(audited.get("question"))].append(
            audited
        )
    for duplicate_group in normalized_groups.values():
        if len(duplicate_group) <= 1:
            continue
        canonical = min(
            duplicate_group,
            key=lambda row: str(row["candidate_id"]),
        )
        canonical["internal_overlap_status"] = "canonical_duplicate"
        for duplicate in duplicate_group:
            if duplicate is canonical:
                continue
            duplicate["internal_overlap_status"] = "rejected"
            duplicate["internal_duplicate_of"] = canonical["candidate_id"]
            duplicate["overlap_decision"] = "reject"
            if "exact_question_internal" not in duplicate["overlap_reasons"]:
                duplicate["overlap_reasons"].append("exact_question_internal")

    active_candidates = [
        row
        for row in audited_candidates
        if row["overlap_decision"] != "reject"
    ]
    for left, right in itertools.combinations(active_candidates, 2):
        if left["independence_unit_id"] == right["independence_unit_id"]:
            continue
        similarity = jaccard_similarity(
            left.get("question"),
            right.get("question"),
            ngram_size,
        )
        for current, compared in ((left, right), (right, left)):
            if similarity > current["max_internal_question_similarity"]:
                current["max_internal_question_similarity"] = round(
                    similarity,
                    6,
                )
                current["max_internal_similar_candidate_id"] = compared[
                    "candidate_id"
                ]
        if similarity < threshold:
            continue
        for current in (left, right):
            current["internal_overlap_status"] = "needs_review"
            current["overlap_decision"] = "needs_review"
            if "near_duplicate_internal" not in current["overlap_reasons"]:
                current["overlap_reasons"].append("near_duplicate_internal")
        review_queue.append(
            {
                "review_id": (
                    f"OVR-{left['candidate_id']}-{right['candidate_id']}"
                ),
                "candidate_id": left["candidate_id"],
                "comparison_scope": "candidate_internal",
                "compared_id": right["candidate_id"],
                "similarity": round(similarity, 6),
                "candidate_question": left.get("question", ""),
                "compared_question": right.get("question", ""),
                "manual_decision": "",
                "reviewer_id": "",
                "reviewed_at": "",
                "review_reason": "",
            }
        )
    rejected_ids = {
        row["candidate_id"]
        for row in audited_candidates
        if row["overlap_decision"] == "reject"
    }
    review_queue = [
        row for row in review_queue if row["candidate_id"] not in rejected_ids
    ]
    return {
        "audited_candidates": audited_candidates,
        "review_queue": review_queue,
    }


def select_deduplicated_candidates(
    result: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    review_queue = result.get("review_queue", [])
    unresolved = [
        row
        for row in review_queue
        if row.get("manual_decision") not in {"keep", "reject"}
    ]
    if unresolved and config.get("fail_on_unresolved_review", True):
        raise ValueError(
            f"仍有 {len(unresolved)} 条近重复等待人工复核，不能生成去重候选池"
        )
    rejected_by_review = {
        row["candidate_id"]
        for row in review_queue
        if row.get("manual_decision") == "reject"
    }
    selected: list[dict[str, Any]] = []
    for audited in result.get("audited_candidates", []):
        if (
            audited.get("overlap_decision") == "reject"
            or audited.get("candidate_id") in rejected_by_review
        ):
            continue
        if audited.get("overlap_decision") == "needs_review" and unresolved:
            continue
        row = deepcopy(audited)
        row["overlap_decision"] = "keep"
        row["candidate_status"] = config["audited_candidate_status"]
        row["overlap_audit_version"] = config["audit_version"]
        row["dataset_version"] = config["output_dataset_version"]
        selected.append(row)
    return sorted(selected, key=lambda row: str(row["candidate_id"]))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=REVIEW_QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in REVIEW_QUEUE_FIELDS}
            for row in sorted(rows, key=lambda item: item["review_id"])
        )


def _build_audit_report(
    *,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    result: dict[str, Any],
    config: dict[str, Any],
    input_hashes: dict[str, str],
    output_hashes: dict[str, str],
) -> dict[str, Any]:
    audited = result["audited_candidates"]
    review_queue = result["review_queue"]
    rejected = [
        row for row in audited if row["overlap_decision"] == "reject"
    ]
    unresolved = [
        row for row in review_queue if not row.get("manual_decision")
    ]
    independence_units = {
        row["independence_unit_id"] for row in selected
    }
    unit_sizes = Counter(
        row["independence_unit_id"] for row in selected
    )
    group_size_distribution = Counter(unit_sizes.values())
    return {
        "audit_version": config["audit_version"],
        "config_version": config["config_version"],
        "dataset_version": config["output_dataset_version"],
        "schema_version": config["schema_version"],
        "protocol_version": config["protocol_version"],
        "kb_version": config["kb_version"],
        "audit_status": "complete" if not unresolved else "blocked_review",
        "counts": {
            "input_candidates": len(candidates),
            "kept_candidates": len(selected),
            "rejected_candidates": len(rejected),
            "review_queue": len(review_queue),
            "unresolved_review": len(unresolved),
            "independence_units": len(independence_units),
        },
        "similarity": {
            "method": "character_ngram_jaccard",
            "ngram_size": int(config["ngram_size"]),
            "jaccard_threshold": float(config["jaccard_threshold"]),
            "normalization_rules": list(config["normalization_rules"]),
        },
        "grouping": {
            "fields": list(config["grouping_fields"]),
            "same_group_is_independent": False,
            "group_size_distribution": {
                str(size): count
                for size, count in sorted(group_size_distribution.items())
            },
        },
        "status_distribution": {
            "dev50_overlap_status": dict(
                sorted(Counter(row["dev50_overlap_status"] for row in audited).items())
            ),
            "internal_overlap_status": dict(
                sorted(Counter(row["internal_overlap_status"] for row in audited).items())
            ),
            "overlap_decision": dict(
                sorted(Counter(row["overlap_decision"] for row in audited).items())
            ),
        },
        "rejected_records": [
            {
                "candidate_id": row["candidate_id"],
                "reasons": list(row["overlap_reasons"]),
            }
            for row in rejected
        ],
        "manual_review": {
            "allowed_decisions": ["keep", "reject"],
            "conclusions": [
                {
                    "review_id": row["review_id"],
                    "manual_decision": row["manual_decision"],
                    "reviewer_id": row["reviewer_id"],
                    "reviewed_at": row["reviewed_at"],
                    "review_reason": row["review_reason"],
                }
                for row in review_queue
                if row.get("manual_decision")
            ],
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
    }


def run_overlap_audit(
    *,
    candidates_path: str | Path,
    dev50_path: str | Path,
    registry_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    candidates_path = Path(candidates_path)
    dev50_path = Path(dev50_path)
    registry_path = Path(registry_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_config(config_path)
    candidates = _load_jsonl(candidates_path)
    dev50_rows = _load_jsonl(dev50_path)
    _validate_inputs(candidates, dev50_rows, config)
    result = audit_candidate_overlap(
        candidates,
        dev50_rows,
        load_dev50_registry(registry_path),
        config,
    )
    selected = select_deduplicated_candidates(result, config)
    if len(
        {row["independence_unit_id"] for row in selected}
    ) != int(config["expected_independence_unit_count"]):
        raise ValueError("独立性单元数量与配置不一致")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_output = output_dir / OUTPUT_CANDIDATES
    queue_output = output_dir / OUTPUT_REVIEW_QUEUE
    audit_output = output_dir / OUTPUT_AUDIT
    _write_jsonl(candidate_output, selected)
    _write_review_queue(queue_output, result["review_queue"])
    report = _build_audit_report(
        candidates=candidates,
        selected=selected,
        result=result,
        config=config,
        input_hashes={
            "candidates": _compute_sha256(candidates_path),
            "dev50": _compute_sha256(dev50_path),
            "dev50_registry": _compute_sha256(registry_path),
            "config": _compute_sha256(config_path),
        },
        output_hashes={
            "deduplicated_candidates": _compute_sha256(candidate_output),
            "review_queue": _compute_sha256(queue_output),
        },
    )
    audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def run_revised_question_reaudit(
    *,
    candidate_id: str,
    pass1_queue_path: str | Path,
    dev50_path: str | Path,
    config_path: str | Path,
    parent_audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    pass1_queue_path = Path(pass1_queue_path)
    dev50_path = Path(dev50_path)
    config_path = Path(config_path)
    parent_audit_path = Path(parent_audit_path)
    output_path = Path(output_path)
    with pass1_queue_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"修订题 candidate_id 必须唯一存在于首轮队列: {candidate_id}")
    report = audit_revised_question_overlap(
        matches[0],
        rows,
        _load_jsonl(dev50_path),
        load_config(config_path),
    )
    report["parent_artifacts"] = {
        "queue_path": pass1_queue_path.as_posix(),
        "queue_sha256": _compute_sha256(pass1_queue_path),
        "overlap_audit_path": parent_audit_path.as_posix(),
        "overlap_audit_sha256": _compute_sha256(parent_audit_path),
        "config_path": config_path.as_posix(),
        "config_sha256": _compute_sha256(config_path),
        "dev50_path": dev50_path.as_posix(),
        "dev50_sha256": _compute_sha256(dev50_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def _default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_root = repo_root / "revision" / "benchmark"
    return {
        "candidates_path": (
            benchmark_root
            / "benchmark_v1"
            / "benchmark_candidates_v0_1.jsonl"
        ),
        "dev50_path": benchmark_root / "dev50" / "dev50_v1_0_frozen.jsonl",
        "registry_path": benchmark_root / "dev50" / "evidence_anchor_registry.md",
        "config_path": (
            repo_root
            / "experiments"
            / "phase7_formal_experiments"
            / "configs"
            / "benchmark_overlap_audit_v0_1.json"
        ),
        "output_dir": benchmark_root / "benchmark_v1",
    }


def main() -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Audit Benchmark-v1 candidates for Dev50 and internal overlap."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "revision"],
        default="full",
    )
    for argument, default in defaults.items():
        parser.add_argument(
            f"--{argument.replace('_', '-')}",
            type=Path,
            default=default,
        )
    parser.add_argument("--candidate-id")
    parser.add_argument("--pass1-queue", type=Path)
    parser.add_argument("--parent-audit", type=Path)
    parser.add_argument("--reaudit-output", type=Path)
    args = parser.parse_args()
    if args.mode == "full":
        report = run_overlap_audit(
            candidates_path=args.candidates_path,
            dev50_path=args.dev50_path,
            registry_path=args.registry_path,
            config_path=args.config_path,
            output_dir=args.output_dir,
        )
        printable = report["counts"]
    else:
        required = {
            "candidate-id": args.candidate_id,
            "pass1-queue": args.pass1_queue,
            "parent-audit": args.parent_audit,
            "reaudit-output": args.reaudit_output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error("revision 模式缺少参数: " + ", ".join(missing))
        report = run_revised_question_reaudit(
            candidate_id=args.candidate_id,
            pass1_queue_path=args.pass1_queue,
            dev50_path=args.dev50_path,
            config_path=args.config_path,
            parent_audit_path=args.parent_audit,
            output_path=args.reaudit_output,
        )
        printable = {
            "candidate_id": report["candidate_id"],
            "reaudit_decision": report["reaudit_decision"],
        }
    print(json.dumps(printable, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
