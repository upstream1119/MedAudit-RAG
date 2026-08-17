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


NON_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+")
PROMOTABLE_REVIEW_FIELDS = (
    "author_final_question",
    "author_final_decision",
    "author_final_risk_labels",
    "author_allowed_answer_scope",
    "author_forbidden_claims",
)
IMMUTABLE_LINK_FIELDS = (
    "question",
    "provisional_expected_decision",
    "source_id",
    "page_number",
    "independence_unit_id",
)


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


def _parse_nonempty_json_list(value: Any, field: str, candidate_id: str) -> list:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value or ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{candidate_id} invalid {field}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{candidate_id} invalid {field}")
    if any(not _normalize_text(item) for item in parsed):
        raise ValueError(f"{candidate_id} invalid {field}")
    return parsed


def _page(value: Any) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid page_number: {value}") from exc
    if page <= 0:
        raise ValueError(f"invalid page_number: {value}")
    return page


def _validate_unique_ids(
    rows: list[dict[str, Any]], field: str, expected_count: int
) -> dict[str, dict[str, Any]]:
    values = [_normalize_text(row.get(field)) for row in rows]
    if len(rows) != expected_count or len(set(values)) != expected_count or "" in values:
        raise ValueError(f"{field} must cover all {expected_count} candidates exactly once")
    return {value: row for value, row in zip(values, rows)}


def _validate_immutable_link(
    review: dict[str, Any], candidate: dict[str, Any]
) -> None:
    candidate_id = str(candidate["candidate_id"])
    for field in IMMUTABLE_LINK_FIELDS:
        left: Any = review.get(field)
        right: Any = candidate.get(field)
        if field == "page_number":
            left = _page(left)
            right = _page(right)
        else:
            left = _normalize_text(left)
            right = _normalize_text(right)
        if left != right:
            raise ValueError(f"{candidate_id} immutable field drift: {field}")


def _validate_rejected_row(review: dict[str, Any]) -> None:
    candidate_id = str(review["candidate_id"])
    leaked = [
        field
        for field in PROMOTABLE_REVIEW_FIELDS
        if _normalize_text(review.get(field))
    ]
    if leaked:
        raise ValueError(
            f"{candidate_id} rejected row carries promotable fields: {', '.join(leaked)}"
        )
    if not _normalize_text(review.get("author_reason")):
        raise ValueError(f"{candidate_id} rejected row requires author_reason")


def _reviewed_candidate(
    review: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    final_question = _normalize_text(review.get("author_final_question"))
    final_decision = _normalize_text(review.get("author_final_decision"))
    if not final_question:
        raise ValueError(f"{candidate_id} invalid author_final_question")
    if final_decision not in set(config["allowed_final_decisions"]):
        raise ValueError(f"{candidate_id} invalid author_final_decision")
    risk_labels = _parse_nonempty_json_list(
        review.get("author_final_risk_labels"),
        "author_final_risk_labels",
        candidate_id,
    )
    forbidden_claims = _parse_nonempty_json_list(
        review.get("author_forbidden_claims"),
        "author_forbidden_claims",
        candidate_id,
    )
    allowed_scope = _normalize_text(review.get("author_allowed_answer_scope"))
    author_reason = _normalize_text(review.get("author_reason"))
    reviewer_id = _normalize_text(review.get("reviewer_id"))
    reviewed_at = _normalize_text(review.get("reviewed_at"))
    for field, value in (
        ("author_allowed_answer_scope", allowed_scope),
        ("author_reason", author_reason),
        ("reviewer_id", reviewer_id),
        ("reviewed_at", reviewed_at),
    ):
        if not value:
            raise ValueError(f"{candidate_id} invalid {field}")

    output = dict(candidate)
    output.update(
        {
            "pre_review_question": candidate["question"],
            "question": final_question,
            "author_outcome": "accepted",
            "reviewed_expected_decision": final_decision,
            "reviewed_risk_labels": risk_labels,
            "reviewed_allowed_answer_scope": allowed_scope,
            "reviewed_forbidden_claims": forbidden_claims,
            "author_reason": author_reason,
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "candidate_status": "author_reviewed_candidate",
            "annotation_status": "author_review_complete",
            "freeze_status": "draft",
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
        }
    )
    return output


def build_reviewed_pool(
    review_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_review_count = int(config["expected_review_count"])
    candidates_by_id = _validate_unique_ids(
        candidate_rows, "candidate_id", expected_review_count
    )
    reviews_by_id = _validate_unique_ids(
        review_rows, "candidate_id", expected_review_count
    )
    if set(candidates_by_id) != set(reviews_by_id):
        raise ValueError(
            f"review rows must cover all {expected_review_count} candidates exactly once"
        )

    allowed_outcomes = set(config["allowed_author_outcomes"])
    accepted: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    for candidate_id in sorted(candidates_by_id):
        candidate = candidates_by_id[candidate_id]
        review = reviews_by_id[candidate_id]
        _validate_immutable_link(review, candidate)
        outcome = _normalize_text(review.get("author_outcome"))
        if outcome not in allowed_outcomes:
            raise ValueError(
                f"{candidate_id} unsupported author outcome: {outcome or '<blank>'}"
            )
        outcome_counts[outcome] += 1
        if outcome == "accepted":
            accepted.append(_reviewed_candidate(review, candidate, config))
        elif outcome == "rejected":
            _validate_rejected_row(review)
        else:
            raise ValueError(f"{candidate_id} unsupported author outcome: {outcome}")

    expected_outcomes = {
        "accepted": int(config["expected_accepted_count"]),
        "rejected": int(config["expected_rejected_count"]),
    }
    if dict(sorted(outcome_counts.items())) != dict(sorted(expected_outcomes.items())):
        raise ValueError(
            f"author outcome distribution drift: {dict(sorted(outcome_counts.items()))}"
        )
    decision_counts = Counter(
        row["reviewed_expected_decision"] for row in accepted
    )
    expected_decisions = {
        key: int(value)
        for key, value in config["expected_decision_distribution"].items()
        if int(value) > 0
    }
    if dict(sorted(decision_counts.items())) != dict(sorted(expected_decisions.items())):
        raise ValueError(
            f"reviewed decision distribution drift: {dict(sorted(decision_counts.items()))}"
        )
    return accepted


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
        source_id = _normalize_text(row.get("source_id"))
        page = row.get("page_number", row.get("page"))
        if source_id and page not in (None, ""):
            pages.add((source_id, _page(page)))
        for evidence in row.get("gold_evidence") or []:
            evidence_source = _normalize_text(evidence.get("source_id"))
            evidence_page = evidence.get("page")
            if evidence_source and evidence_page not in (None, ""):
                pages.add((evidence_source, _page(evidence_page)))
    return pages


def _fact_clusters(rows: list[dict[str, Any]]) -> set[str]:
    return {
        cluster
        for row in rows
        for cluster in (
            _normalize_text(
                row.get("provisional_fact_cluster_id", row.get("fact_cluster_id"))
            ),
        )
        if cluster
    }


def audit_reviewed_pool(
    candidates: list[dict[str, Any]],
    *,
    dev50_rows: list[dict[str, Any]],
    frozen15_ids: set[str],
    existing_candidates: list[dict[str, Any]],
    ngram_size: int,
    threshold: float,
) -> dict[str, Any]:
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    fact_clusters = [
        _normalize_text(row.get("provisional_fact_cluster_id"))
        for row in candidates
    ]
    independence_units = [
        _normalize_text(row.get("independence_unit_id")) for row in candidates
    ]
    source_pages = [
        (_normalize_text(row.get("source_id")), _page(row.get("page_number")))
        for row in candidates
    ]
    structural_sets = {
        "candidate_id": candidate_ids,
        "provisional_fact_cluster_id": fact_clusters,
        "independence_unit_id": independence_units,
        "source_page": source_pages,
    }
    invalid_structures = [
        name
        for name, values in structural_sets.items()
        if any(value in ("", ("", 0)) for value in values)
        or len(set(values)) != len(values)
    ]
    if invalid_structures:
        raise ValueError(
            "structural independence failed: " + ", ".join(invalid_structures)
        )

    existing_pages = _source_pages(dev50_rows) | _source_pages(existing_candidates)
    dev_clusters = _fact_clusters(dev50_rows)
    existing_clusters = _fact_clusters(existing_candidates)
    review_queue: list[dict[str, Any]] = []
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
        peers = [
            {"candidate_id": other["candidate_id"], "question": other["question"]}
            for peer_index, other in enumerate(candidates)
            if peer_index != index
        ]
        internal_match = _best_question_match(
            candidate["question"],
            peers,
            id_field="candidate_id",
            ngram_size=ngram_size,
        )
        source_page = (
            str(candidate["source_id"]),
            _page(candidate["page_number"]),
        )
        fact_cluster = str(candidate["provisional_fact_cluster_id"])
        reasons: list[str] = []
        if float(dev_match["similarity"]) >= threshold:
            reasons.append("dev50_question_overlap")
        if float(existing_match["similarity"]) >= threshold:
            reasons.append("existing_candidate_question_overlap")
        if float(internal_match["similarity"]) >= threshold:
            reasons.append("internal_candidate_question_overlap")
        if source_page in existing_pages:
            reasons.append("source_page_overlap")
        if fact_cluster in dev_clusters:
            reasons.append("dev50_fact_cluster_overlap")
        if fact_cluster in existing_clusters:
            reasons.append("existing_candidate_fact_cluster_overlap")
        if reasons:
            review_queue.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reasons": reasons,
                    "dev50_match": dev_match,
                    "frozen15_overlap": (
                        float(dev_match["similarity"]) >= threshold
                        and str(dev_match["id"]) in frozen15_ids
                    ),
                    "existing_candidate_match": existing_match,
                    "internal_candidate_match": internal_match,
                    "source_page": list(source_page),
                    "fact_cluster_id": fact_cluster,
                }
            )
    if review_queue:
        raise ValueError(
            f"unresolved overlap found for {len(review_queue)} reviewed candidates"
        )

    return {
        "status": "reviewed_candidate_pool_ready",
        "candidate_count": len(candidates),
        "unique_candidate_count": len(set(candidate_ids)),
        "unique_fact_cluster_count": len(set(fact_clusters)),
        "unique_independence_unit_count": len(set(independence_units)),
        "unique_source_page_count": len(set(source_pages)),
        "decision_distribution": dict(
            sorted(
                Counter(
                    str(row["reviewed_expected_decision"])
                    for row in candidates
                ).items()
            )
        ),
        "dev50_overlap_count": 0,
        "frozen15_overlap_count": 0,
        "existing_candidate_overlap_count": 0,
        "internal_candidate_overlap_count": 0,
        "source_page_overlap_count": 0,
        "fact_cluster_overlap_count": 0,
        "unresolved_overlap_count": 0,
        "review_queue": [],
        "candidate_pool_materialized": True,
        "benchmark_merge_performed": False,
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "usage": zero_usage(),
    }


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_input_hashes(config: dict[str, Any], root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_hash in config["input_sha256"].items():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"locked input is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"locked input drift: {relative_path}, "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        observed[relative_path] = actual_hash
    return observed


def _validate_parent_review_audit(
    audit: dict[str, Any], config: dict[str, Any]
) -> None:
    expected_outcomes = {
        "accepted": int(config["expected_accepted_count"]),
        "rejected": int(config["expected_rejected_count"]),
    }
    if audit.get("status") != "author_review_complete":
        raise ValueError("parent author review is not complete")
    if int(audit.get("author_reviewed_count", -1)) != int(
        config["expected_review_count"]
    ):
        raise ValueError("parent author review count drift")
    if audit.get("outcome_counts") != expected_outcomes:
        raise ValueError("parent author outcome distribution drift")
    for field in (
        "candidate_merge_performed",
        "gold_promotion_performed",
        "freeze_performed",
    ):
        if audit.get(field) is not False:
            raise ValueError(f"parent author review boundary drift: {field}")


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


def _write_summary(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Benchmark-v1 补充已审核候选池 v0.2",
        "",
        f"- 状态：`{audit['status']}`",
        f"- 已审核候选：{audit['candidate_count']}",
        f"- 排除的作者拒绝项：{audit['excluded_rejected_count']}",
        f"- 最终题面修订：{audit['question_revision_count']}",
        f"- 真实决策分布：`{audit['decision_distribution']}`",
        f"- 唯一事实簇：{audit['unique_fact_cluster_count']}",
        f"- 唯一独立单元：{audit['unique_independence_unit_count']}",
        f"- 唯一来源页：{audit['unique_source_page_count']}",
        f"- 未解决重叠：{audit['unresolved_overlap_count']}",
        "- Benchmark 合并：否",
        "- Gold 晋升：否",
        "- Benchmark 冻结：否",
        "- 外部模型/API 调用：0",
        "",
        audit["medical_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(config_path: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _read_json(config_path)
    observed_hashes = _validate_input_hashes(config, root)
    review_rows = _read_csv(root / config["review_csv_path"])
    candidate_rows = _read_jsonl(root / config["candidate_path"])
    parent_audit = _read_json(root / config["review_audit_path"])
    _validate_parent_review_audit(parent_audit, config)
    reviewed_pool = build_reviewed_pool(review_rows, candidate_rows, config)
    dev50_rows = _read_jsonl(root / config["dev50_path"])
    existing_candidates = [
        row
        for relative_path in config["existing_candidate_paths"]
        for row in _read_jsonl(root / relative_path)
    ]
    audit = audit_reviewed_pool(
        reviewed_pool,
        dev50_rows=dev50_rows,
        frozen15_ids=set(config["frozen15_sample_ids"]),
        existing_candidates=existing_candidates,
        ngram_size=int(config["ngram_size"]),
        threshold=float(config["jaccard_threshold"]),
    )
    if audit["candidate_count"] != int(config["expected_accepted_count"]):
        raise ValueError("reviewed candidate count drift")
    expected_decisions = {
        key: int(value)
        for key, value in config["expected_decision_distribution"].items()
        if int(value) > 0
    }
    if audit["decision_distribution"] != expected_decisions:
        raise ValueError("reviewed decision distribution drift")

    rejected_ids = sorted(
        str(row["candidate_id"])
        for row in review_rows
        if row["author_outcome"] == "rejected"
    )
    audit.update(
        {
            "config_version": config["config_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "reviewed_count": len(review_rows),
            "accepted_count": len(reviewed_pool),
            "excluded_rejected_count": len(rejected_ids),
            "excluded_rejected_candidate_ids": rejected_ids,
            "question_revision_count": sum(
                row["question"] != row["pre_review_question"]
                for row in reviewed_pool
            ),
            "input_sha256": observed_hashes,
            "medical_boundary": (
                "该产物只是经过单一作者审核并重新完成独立性审计的候选池；"
                "它不等于 Gold 标签、Benchmark 合并、冻结测试集、独立专家验证"
                "或临床验证。"
            ),
        }
    )
    output_dir = root / config["output_dir"]
    candidate_output = output_dir / config["candidate_output"]
    audit_output = output_dir / config["audit_output"]
    summary_output = output_dir / config["summary_output"]
    _write_jsonl(candidate_output, reviewed_pool)
    audit["candidate_output_sha256"] = _sha256(candidate_output)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_summary(summary_output, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and re-audit the accepted supplement candidate pool."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark_anchor_supplement_reviewed_pool_v0_2.json"
        ),
    )
    args = parser.parse_args()
    audit = run(args.config)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
