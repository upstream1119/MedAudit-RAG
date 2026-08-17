from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {"review_required", "boundary_refusal"}
REQUIRED_ANCHOR_FIELDS = {
    "anchor_id",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "page_number",
    "text_span",
    "supported_claim_types",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
    "scope_check",
    "verification_status",
}
NON_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+")
GARBLED_TEXT_RE = re.compile(r"\?{2,}|�")


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


def _character_ngrams(value: Any, n: int) -> set[str]:
    text = normalize_question(value)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _jaccard(left: Any, right: Any, n: int) -> float:
    left_grams = _character_ngrams(left, n)
    right_grams = _character_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _validate_anchor(anchor: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ANCHOR_FIELDS - set(anchor))
    if missing:
        raise ValueError(
            f"{anchor.get('anchor_id', '<unknown>')} 缺少锚点字段: "
            f"{', '.join(missing)}"
        )
    if anchor["verification_status"] != "author_verified_anchor":
        raise ValueError(f"锚点未经作者核验: {anchor['anchor_id']}")
    if anchor["scope_check"] != "within_can_support":
        raise ValueError(f"锚点超出 can_support 范围: {anchor['anchor_id']}")
    if int(anchor["page_number"]) <= 0:
        raise ValueError(f"锚点页码无效: {anchor['anchor_id']}")
    for field in ("text_span", "evidence_scope", "age_scope"):
        value = _normalize_text(anchor[field])
        if not value or GARBLED_TEXT_RE.search(value):
            raise ValueError(f"锚点字段不可用: {anchor['anchor_id']} {field}")


def _validate_config(config: dict[str, Any]) -> None:
    specs = config.get("challenge_specs")
    if not isinstance(specs, list):
        raise ValueError("配置必须提供 challenge_specs")
    expected_count = int(config.get("expected_candidate_count", -1))
    if len(specs) != expected_count:
        raise ValueError(f"候选数量应为 {expected_count}，实际为 {len(specs)}")
    decisions = Counter(
        str(spec.get("provisional_expected_decision", "")) for spec in specs
    )
    invalid = sorted(set(decisions) - ALLOWED_DECISIONS)
    if invalid:
        raise ValueError(
            "本轮只允许补充 review_required 和 boundary_refusal: "
            + ", ".join(invalid)
        )
    target = Counter(
        {
            key: int(value)
            for key, value in config.get("target_decision_distribution", {}).items()
            if int(value) > 0
        }
    )
    if decisions != target:
        raise ValueError(
            f"候选决策分布与配置不一致: actual={dict(decisions)}, "
            f"target={dict(target)}"
        )
    if int(config.get("external_model_calls", -1)) != 0:
        raise ValueError("补充候选构建不允许外部模型调用")


def build_candidates(
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build draft challenge candidates without promoting provisional labels."""
    _validate_config(config)
    anchors_by_id: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        _validate_anchor(anchor)
        anchor_id = str(anchor["anchor_id"])
        if anchor_id in anchors_by_id:
            raise ValueError(f"锚点 ID 重复: {anchor_id}")
        anchors_by_id[anchor_id] = anchor

    spec_anchor_ids = [str(spec.get("anchor_id", "")) for spec in config["challenge_specs"]]
    duplicate_anchor_ids = sorted(
        anchor_id
        for anchor_id, count in Counter(spec_anchor_ids).items()
        if count > 1
    )
    if duplicate_anchor_ids:
        raise ValueError(
            "补充候选重复使用锚点: " + ", ".join(duplicate_anchor_ids)
        )
    unknown = sorted(set(spec_anchor_ids) - set(anchors_by_id))
    if unknown:
        raise ValueError("补充候选绑定未知锚点: " + ", ".join(unknown))
    unused = sorted(set(anchors_by_id) - set(spec_anchor_ids))
    if unused:
        raise ValueError("存在未生成候选的新锚点: " + ", ".join(unused))

    candidates: list[dict[str, Any]] = []
    questions: set[str] = set()
    policy_rule_id = str(config["policy_rule_id"])
    minimum_chars = int(config.get("minimum_question_chars", 12))
    maximum_chars = int(config.get("maximum_question_chars", 320))
    for spec in config["challenge_specs"]:
        anchor_id = str(spec["anchor_id"])
        anchor = anchors_by_id[anchor_id]
        question = _normalize_text(spec.get("question"))
        normalized_question = normalize_question(question)
        if not minimum_chars <= len(question) <= maximum_chars:
            raise ValueError(f"候选问题长度不合规: {anchor_id}")
        if not normalized_question or GARBLED_TEXT_RE.search(question):
            raise ValueError(f"候选问题为空或含乱码: {anchor_id}")
        if normalized_question in questions:
            raise ValueError(f"补充候选问题重复: {anchor_id}")
        questions.add(normalized_question)

        decision = str(spec["provisional_expected_decision"])
        boundary = decision == "boundary_refusal"
        expected_support = "policy_rule" if boundary else "partial_current_kb_support"
        if str(spec.get("current_kb_support")) != expected_support:
            raise ValueError(f"候选 KB support 与决策不一致: {anchor_id}")
        policy_rule_ids = list(spec.get("policy_rule_ids") or [])
        if boundary and policy_rule_ids != [policy_rule_id]:
            raise ValueError(
                f"boundary_refusal 必须绑定 {policy_rule_id}: {anchor_id}"
            )
        if not boundary and policy_rule_ids:
            raise ValueError(f"review_required 不得绑定处方拒答规则: {anchor_id}")

        candidate_id = _stable_id(
            "PMSQA-BV1X",
            anchor_id,
            question,
            config["config_version"],
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "question": question,
                "language": "zh-CN",
                "candidate_role": "independent_anchor_boundary_challenge",
                "challenge_type": str(spec["challenge_type"]),
                "provisional_expected_decision": decision,
                "provisional_scenario_type": (
                    "prescription-boundary" if boundary else "evidence-scope"
                ),
                "provisional_risk_labels": sorted(
                    set(spec.get("provisional_risk_labels") or [])
                ),
                "current_kb_support": expected_support,
                "missing_evidence_type": list(
                    spec.get("missing_evidence_type") or []
                ),
                "policy_rule_ids": policy_rule_ids,
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
                "evidence_anchor_group_id": _stable_id("EAG-BV1X", anchor_id),
                "provisional_fact_cluster_id": _stable_id("FC-BV1X", anchor_id),
                "independence_unit_id": _stable_id("IU-BV1X", anchor_id),
                "candidate_status": config["candidate_status"],
                "candidate_generation_method": "explicit_author_review_spec",
                "annotation_status": "pending_author_review",
                "freeze_status": "draft",
                "dataset_version": config["dataset_version"],
                "schema_version": config["schema_version"],
                "protocol_version": config["protocol_version"],
                "kb_version": config["kb_version"],
                "generator_version": config["generator_version"],
                "config_version": config["config_version"],
            }
        )
    return sorted(candidates, key=lambda row: row["candidate_id"])


def _best_question_match(
    question: str,
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    ngram_size: int,
) -> dict[str, Any]:
    best = {"id": "", "similarity": 0.0, "exact": False}
    normalized = normalize_question(question)
    for row in rows:
        other = row.get("question", "")
        exact = normalized == normalize_question(other)
        similarity = 1.0 if exact else _jaccard(question, other, ngram_size)
        if similarity > float(best["similarity"]):
            best = {
                "id": str(row.get(id_field, "")),
                "similarity": round(similarity, 6),
                "exact": exact,
            }
    return best


def _source_pages(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for row in rows:
        source_id = str(row.get("source_id", ""))
        page = row.get("page_number", row.get("page"))
        if source_id and page not in (None, ""):
            pages.add((source_id, int(page)))
        for evidence in row.get("gold_evidence") or []:
            evidence_source = str(evidence.get("source_id", ""))
            evidence_page = evidence.get("page")
            if evidence_source and evidence_page not in (None, ""):
                pages.add((evidence_source, int(evidence_page)))
    return pages


def audit_overlap(
    candidates: list[dict[str, Any]],
    *,
    dev50_rows: list[dict[str, Any]],
    frozen15_ids: set[str],
    existing_candidates: list[dict[str, Any]],
    ngram_size: int,
    threshold: float,
) -> dict[str, Any]:
    dev50_overlap_count = 0
    frozen15_overlap_count = 0
    existing_overlap_count = 0
    source_page_overlap_count = 0
    unresolved_ids: set[str] = set()
    review_queue: list[dict[str, Any]] = []
    existing_pages = _source_pages(dev50_rows) | _source_pages(existing_candidates)

    for index, candidate in enumerate(candidates):
        dev_match = _best_question_match(
            candidate["question"],
            dev50_rows,
            id_field="sample_id",
            ngram_size=ngram_size,
        )
        existing_match = _best_question_match(
            candidate["question"],
            existing_candidates,
            id_field="candidate_id",
            ngram_size=ngram_size,
        )
        peer_rows = [
            {
                "candidate_id": other["candidate_id"],
                "question": other["question"],
            }
            for peer_index, other in enumerate(candidates)
            if peer_index != index
        ]
        internal_match = _best_question_match(
            candidate["question"],
            peer_rows,
            id_field="candidate_id",
            ngram_size=ngram_size,
        )
        dev_overlap = float(dev_match["similarity"]) >= threshold
        existing_overlap = float(existing_match["similarity"]) >= threshold
        internal_overlap = float(internal_match["similarity"]) >= threshold
        source_page_overlap = (
            str(candidate["source_id"]), int(candidate["page_number"])
        ) in existing_pages
        frozen_overlap = dev_overlap and str(dev_match["id"]) in frozen15_ids
        if dev_overlap:
            dev50_overlap_count += 1
        if frozen_overlap:
            frozen15_overlap_count += 1
        if existing_overlap:
            existing_overlap_count += 1
        if source_page_overlap:
            source_page_overlap_count += 1
        reasons = []
        if dev_overlap:
            reasons.append("dev50_question_overlap")
        if existing_overlap:
            reasons.append("existing_candidate_question_overlap")
        if internal_overlap:
            reasons.append("internal_candidate_question_overlap")
        if source_page_overlap:
            reasons.append("source_page_overlap")
        if reasons:
            unresolved_ids.add(candidate["candidate_id"])
            review_queue.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "question": candidate["question"],
                    "reasons": reasons,
                    "dev50_match": dev_match,
                    "frozen15_overlap": frozen_overlap,
                    "existing_candidate_match": existing_match,
                    "internal_candidate_match": internal_match,
                    "source_page_overlap": source_page_overlap,
                }
            )
    return {
        "candidate_count": len(candidates),
        "dev50_overlap_count": dev50_overlap_count,
        "frozen15_overlap_count": frozen15_overlap_count,
        "existing_candidate_overlap_count": existing_overlap_count,
        "source_page_overlap_count": source_page_overlap_count,
        "unresolved_overlap_count": len(unresolved_ids),
        "review_queue": review_queue,
        "freeze_performed": False,
        "gold_promotion_performed": False,
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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_input_hashes(config: dict[str, Any], root: Path) -> None:
    for relative_path, expected_hash in config.get("input_sha256", {}).items():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"锁定输入不存在: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"锁定输入发生漂移: {relative_path}, "
                f"expected={expected_hash}, actual={actual_hash}"
            )


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


def _write_review_queue(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "question",
        "provisional_expected_decision",
        "challenge_type",
        "source_id",
        "source_title",
        "page_number",
        "anchor_text_span",
        "evidence_scope",
        "age_scope",
        "applicability_conditions",
        "evidence_anchor_ids",
        "independence_unit_id",
        "author_outcome",
        "author_final_decision",
        "author_reason",
        "reviewer_id",
        "reviewed_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {field: candidate.get(field, "") for field in fields}
            row["evidence_anchor_ids"] = json.dumps(
                candidate["evidence_anchor_ids"], ensure_ascii=False
            )
            writer.writerow(row)


def _write_summary(
    path: Path,
    candidates: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    decisions = Counter(
        row["provisional_expected_decision"] for row in candidates
    )
    lines = [
        "# Benchmark-v1 独立锚点补充候选 v0.2",
        "",
        f"- 候选数：{len(candidates)}",
        f"- 独立单元数：{len({row['independence_unit_id'] for row in candidates})}",
        f"- 暂定决策分布：`{dict(sorted(decisions.items()))}`",
        f"- 未解决重叠：{audit['unresolved_overlap_count']}",
        "- 状态：`pending_author_review`",
        "- Gold 晋升：否",
        "- Benchmark 冻结：否",
        "- 外部模型/API 调用：0",
        "",
        "这些记录只是基于独立证据锚点构建的候选题，必须完成作者核验后才可进入后续选择流程。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _load_json(config_path)
    _validate_input_hashes(config, root)
    anchors = _load_jsonl(root / config["anchor_path"])
    dev50_rows = _load_jsonl(root / config["dev50_path"])
    existing_candidates = [
        row
        for relative_path in config["existing_candidate_paths"]
        for row in _load_jsonl(root / relative_path)
    ]
    candidates = build_candidates(anchors, config)
    audit = audit_overlap(
        candidates,
        dev50_rows=dev50_rows,
        frozen15_ids=set(config["frozen15_sample_ids"]),
        existing_candidates=existing_candidates,
        ngram_size=int(config["ngram_size"]),
        threshold=float(config["jaccard_threshold"]),
    )
    audit.update(
        {
            "config_version": config["config_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "decision_distribution": dict(
                sorted(
                    Counter(
                        row["provisional_expected_decision"] for row in candidates
                    ).items()
                )
            ),
            "source_distribution": dict(
                sorted(Counter(row["source_id"] for row in candidates).items())
            ),
            "lineage": {
                "config_path": Path(config_path).as_posix(),
                "config_sha256": _sha256(config_path),
                "anchor_path": config["anchor_path"],
                "anchor_sha256": _sha256(root / config["anchor_path"]),
                "dev50_path": config["dev50_path"],
                "dev50_sha256": _sha256(root / config["dev50_path"]),
            },
        }
    )
    output_dir = root / config["output_dir"]
    _write_jsonl(output_dir / config["candidate_output"], candidates)
    _write_review_queue(output_dir / config["review_queue_output"], candidates)
    (output_dir / config["audit_output"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(output_dir / config["summary_output"], candidates, audit)
    if audit["unresolved_overlap_count"]:
        raise ValueError(
            f"存在 {audit['unresolved_overlap_count']} 条未解决重叠，候选不得进入作者核验"
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build independent-anchor supplement candidates for Benchmark-v1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs")
        / "benchmark_anchor_supplement_candidates_v0_2.json",
    )
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "decision_distribution": report["decision_distribution"],
                "unresolved_overlap_count": report["unresolved_overlap_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
