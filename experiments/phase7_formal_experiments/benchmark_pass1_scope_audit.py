from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def _effective_question(row: dict[str, Any]) -> str:
    return str(row.get("pass1_final_question") or row.get("question") or "")


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term in text for term in terms)


def audit_pass1_scope(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Conservatively flag scope risks without mutating Pass 1 decisions."""
    promotable_outcomes = set(config["promotable_outcomes"])
    medication_markers = list(config["medication_markers"])
    non_medication_claim_types = set(config["non_medication_claim_types"])
    scope_reject_terms = list(config["scope_reject_terms"])

    rejected_scope_units: set[str] = set()
    for row in rows:
        if str(row.get("pass1_outcome")) != "reject":
            continue
        reason_blob = " ".join(
            [
                str(row.get("pass1_review_reason", "")),
                " ".join(_list_value(row.get("pass1_issues_found"))),
            ]
        )
        if _contains_any(reason_blob, scope_reject_terms):
            rejected_scope_units.add(str(row.get("independence_unit_id", "")))

    flagged_rows: list[dict[str, Any]] = []
    promotable_rows = [
        row for row in rows if str(row.get("pass1_outcome")) in promotable_outcomes
    ]
    decision_counts = Counter(
        str(row.get("pass1_expected_decision", "")) for row in promotable_rows
    )
    for row in promotable_rows:
        question = _effective_question(row)
        claim_types = set(_list_value(row.get("supported_claim_types")))
        evidence_types = set(_list_value(row.get("pass1_required_evidence_type")))
        risk_labels = set(_list_value(row.get("pass1_risk_labels")))
        searchable = " ".join(
            [question, " ".join(claim_types), " ".join(evidence_types), " ".join(risk_labels)]
        )
        explicit_medication_link = _contains_any(searchable, medication_markers)
        reasons: list[str] = []
        non_medication_claim = bool(
            (claim_types | evidence_types).intersection(non_medication_claim_types)
        )
        sibling_scope_conflict = (
            str(row.get("independence_unit_id", "")) in rejected_scope_units
        )
        # Missing a keyword alone is only a heuristic signal. Escalate it only
        # when independent metadata also indicates a likely scope mismatch.
        if not explicit_medication_link and non_medication_claim:
            reasons.extend(
                ["no_explicit_medication_link", "non_medication_claim_type"]
            )
        if not explicit_medication_link and sibling_scope_conflict:
            reasons.extend(
                ["no_explicit_medication_link", "sibling_scope_inconsistency"]
            )
        if reasons:
            flagged_rows.append(
                {
                    "annotation_order": int(row.get("annotation_order") or 0),
                    "candidate_id": str(row.get("candidate_id", "")),
                    "independence_unit_id": str(row.get("independence_unit_id", "")),
                    "source_id": str(row.get("source_id", "")),
                    "source_title": str(row.get("source_title", "")),
                    "page_number": int(row.get("page_number") or 0),
                    "question": question,
                    "pass1_outcome": str(row.get("pass1_outcome", "")),
                    "pass1_expected_decision": str(
                        row.get("pass1_expected_decision", "")
                    ),
                    "flag_reasons": sorted(set(reasons)),
                    "recommended_action": "author_scope_review",
                }
            )

    flagged_rows.sort(key=lambda row: (row["annotation_order"], row["candidate_id"]))
    return {
        "audit_version": config.get("audit_version", "pass1-scope-audit-v0.1"),
        "input_row_count": len(rows),
        "promotable_count": len(promotable_rows),
        "promotable_decision_distribution": dict(sorted(decision_counts.items())),
        "flagged_count": len(flagged_rows),
        "flag_reason_distribution": dict(
            sorted(Counter(reason for row in flagged_rows for reason in row["flag_reasons"]).items())
        ),
        "flagged_rows": flagged_rows,
        "mutation_policy": "report_only_no_pass1_mutation",
        "usage": zero_usage(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_queue(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _write_review_queue(path: Path, flagged_rows: list[dict[str, Any]]) -> None:
    fields = [
        "annotation_order",
        "candidate_id",
        "independence_unit_id",
        "source_id",
        "source_title",
        "page_number",
        "question",
        "pass1_outcome",
        "pass1_expected_decision",
        "flag_reasons",
        "recommended_action",
        "author_decision",
        "author_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in flagged_rows:
            payload = dict(row)
            payload["flag_reasons"] = json.dumps(
                row["flag_reasons"], ensure_ascii=False, sort_keys=True
            )
            payload["author_decision"] = ""
            payload["author_reason"] = ""
            writer.writerow(payload)


def run_scope_audit(
    queue_path: str | Path,
    config_path: str | Path,
    report_path: str | Path,
    review_queue_path: str | Path,
) -> dict[str, Any]:
    queue = Path(queue_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    report = audit_pass1_scope(_read_queue(queue), config)
    report["input_path"] = queue.as_posix()
    report["input_sha256"] = _sha256(queue)
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_review_queue(Path(review_queue_path), report["flagged_rows"])
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Audit Pass 1 pediatric medication scope.")
    parser.add_argument(
        "--queue",
        default=root / "revision/benchmark/benchmark_v1/annotation_pass1_queue_v0_1.csv",
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("configs") / "benchmark_pass1_scope_audit_v0_1.json",
    )
    parser.add_argument(
        "--report",
        default=root / "revision/benchmark/benchmark_v1/pass1_scope_consistency_audit_v0_1.json",
    )
    parser.add_argument(
        "--review-queue",
        default=root / "revision/benchmark/benchmark_v1/pass1_scope_consistency_review_queue_v0_1.csv",
    )
    args = parser.parse_args()
    report = run_scope_audit(args.queue, args.config, args.report, args.review_queue)
    print(json.dumps({"promotable": report["promotable_count"], "flagged": report["flagged_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
