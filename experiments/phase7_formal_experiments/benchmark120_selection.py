from __future__ import annotations

import argparse
import csv
import json
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DECISIONS = (
    "answer",
    "review_required",
    "insufficient_evidence",
    "boundary_refusal",
)
NON_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_hashes(
    expected_hashes: dict[str, str],
    *,
    base_dir: str | Path,
) -> dict[str, str]:
    actual: dict[str, str] = {}
    base = Path(base_dir)
    for raw_path, expected in sorted(expected_hashes.items()):
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            raise ValueError(f"parent asset is missing: {raw_path}")
        digest = file_sha256(path)
        actual[raw_path] = digest
        if digest.lower() != str(expected).lower():
            raise ValueError(
                f"parent asset hash mismatch: {raw_path}; "
                f"expected={expected}, actual={digest}"
            )
    return actual


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


def _atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(target)


def _write_json(path: str | Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value or "[]"))
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON list")
    return parsed


def _question_ngrams(value: Any, n: int) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = NON_ALNUM_CJK_RE.sub("", text)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _decision_vector(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    counts = Counter(_text(row.get("expected_decision")) for row in rows)
    return tuple(counts[decision] for decision in DECISIONS)


def _target_vector(distribution: dict[str, int]) -> tuple[int, ...]:
    unknown = set(distribution) - set(DECISIONS)
    if unknown:
        raise ValueError(f"unsupported decisions: {sorted(unknown)}")
    return tuple(int(distribution.get(decision, 0)) for decision in DECISIONS)


def _component_seed_key(rows: list[dict[str, Any]], seed: int) -> str:
    candidate_ids = "|".join(sorted(_text(row.get("candidate_id")) for row in rows))
    return hashlib.sha256(f"{seed}|{candidate_ids}".encode("utf-8")).hexdigest()


def _build_independence_components(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    ngram_size: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _text(row.get("candidate_id")))
    if not ordered:
        return []
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(ordered):
        keys = (
            ("fact", _text(row.get("fact_cluster_id"))),
            ("anchor", _text(row.get("evidence_anchor_group_id"))),
            (
                "source_page",
                _text(row.get("source_id")),
                int(row.get("page_number")),
            ),
        )
        for key in keys:
            if not all(_text(value) for value in key[1:]):
                raise ValueError(f"incomplete independence key: {key[0]}")
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    grams = [_question_ngrams(row.get("question"), ngram_size) for row in ordered]
    for left in range(len(ordered)):
        for right in range(left):
            if _jaccard(grams[left], grams[right]) >= similarity_threshold:
                union(left, right)

    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(ordered):
        grouped[find(index)].append(row)

    components: list[dict[str, Any]] = []
    for component_rows in grouped.values():
        source_pages = {
            (_text(row.get("source_id")), int(row.get("page_number")))
            for row in component_rows
        }
        components.append(
            {
                "rows": sorted(
                    component_rows,
                    key=lambda row: _text(row.get("candidate_id")),
                ),
                "vector": _decision_vector(component_rows),
                "two_pass_count": sum(
                    int(row.get("annotation_pass_count", 0)) >= 2
                    for row in component_rows
                ),
                "source_page_count": len(source_pages),
                "seed_key": _component_seed_key(component_rows, seed),
            }
        )
    return sorted(components, key=lambda component: component["seed_key"])


def _within_target(state: tuple[int, ...], target: tuple[int, ...]) -> bool:
    return all(value <= limit for value, limit in zip(state, target))


def select_benchmark120(
    rows: list[dict[str, Any]],
    *,
    target_distribution: dict[str, int],
    seed: int,
    ngram_size: int,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在证据独立组件边界内选择满足精确决策配额的 120 条草案。"""
    target = _target_vector(target_distribution)
    components = _build_independence_components(
        rows,
        seed=seed,
        ngram_size=ngram_size,
        similarity_threshold=similarity_threshold,
    )
    states: dict[tuple[int, ...], tuple[tuple[int, int, int], int]] = {
        (0, 0, 0, 0): ((0, 0, 0), 0)
    }
    for index, component in enumerate(components):
        updated = dict(states)
        vector = component["vector"]
        for state, (score, mask) in states.items():
            next_state = tuple(
                value + addition for value, addition in zip(state, vector)
            )
            if not _within_target(next_state, target):
                continue
            next_score = (
                score[0] + component["two_pass_count"],
                score[1] + component["source_page_count"],
                score[2] + 1,
            )
            candidate = (next_score, mask | (1 << index))
            if next_state not in updated or candidate[0] > updated[next_state][0]:
                updated[next_state] = candidate
        states = updated
    if target not in states:
        raise ValueError("exact Benchmark120 decision quotas are not reachable")

    score, selected_mask = states[target]
    selected = [
        row
        for index, component in enumerate(components)
        if selected_mask & (1 << index)
        for row in component["rows"]
    ]
    selected = sorted(selected, key=lambda row: _text(row.get("candidate_id")))
    metadata = {
        "selected_count": len(selected),
        "decision_distribution": dict(
            sorted(Counter(row["expected_decision"] for row in selected).items())
        ),
        "selected_two_pass_count": score[0],
        "selected_single_pass_count": len(selected) - score[0],
        "selected_component_count": score[2],
        "selected_source_page_count": score[1],
        "selection_seed": seed,
    }
    return selected, metadata


def propose_grouped_split(
    selected_rows: list[dict[str, Any]],
    *,
    split_targets: dict[str, dict[str, int]],
    desired_validation_two_pass: int,
    seed: int,
    ngram_size: int,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按独立组件提出 Validation/Pilot Test 草案，不执行冻结。"""
    validation_target = _target_vector(split_targets["validation"])
    pilot_target = _target_vector(split_targets["pilot_test"])
    total_target = tuple(
        left + right for left, right in zip(validation_target, pilot_target)
    )
    if _decision_vector(selected_rows) != total_target:
        raise ValueError("selected rows do not match split target totals")

    components = _build_independence_components(
        selected_rows,
        seed=seed,
        ngram_size=ngram_size,
        similarity_threshold=similarity_threshold,
    )
    states: dict[
        tuple[int, ...], dict[int, tuple[int, int]]
    ] = {(0, 0, 0, 0): {0: (0, 0)}}
    for index, component in enumerate(components):
        updated = {state: dict(options) for state, options in states.items()}
        vector = component["vector"]
        for state, options in states.items():
            next_state = tuple(
                value + addition for value, addition in zip(state, vector)
            )
            if not _within_target(next_state, validation_target):
                continue
            next_options = updated.setdefault(next_state, {})
            for two_pass_count, (source_page_count, mask) in options.items():
                next_two_pass = two_pass_count + component["two_pass_count"]
                candidate = (
                    source_page_count + component["source_page_count"],
                    mask | (1 << index),
                )
                existing = next_options.get(next_two_pass)
                if existing is None or candidate[0] > existing[0]:
                    next_options[next_two_pass] = candidate
        states = updated
    if validation_target not in states:
        raise ValueError("exact Validation quotas are not reachable")

    options = states[validation_target]
    validation_two_pass = min(
        options,
        key=lambda count: (
            abs(count - desired_validation_two_pass),
            -options[count][0],
            count,
        ),
    )
    _, validation_mask = options[validation_two_pass]
    split_rows: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        split_name = (
            "validation" if validation_mask & (1 << index) else "pilot_test"
        )
        for row in component["rows"]:
            output = dict(row)
            output["dataset_split"] = split_name
            output["split_status"] = "proposal"
            output["freeze_status"] = "draft"
            split_rows.append(output)
    split_rows.sort(key=lambda row: _text(row.get("candidate_id")))

    decision_distribution = {
        split_name: dict(
            sorted(
                Counter(
                    row["expected_decision"]
                    for row in split_rows
                    if row["dataset_split"] == split_name
                ).items()
            )
        )
        for split_name in ("validation", "pilot_test")
    }
    actual_pilot = _target_vector(decision_distribution["pilot_test"])
    if actual_pilot != pilot_target:
        raise ValueError("Pilot Test quotas drifted after grouped split")
    split_counts = dict(
        sorted(Counter(row["dataset_split"] for row in split_rows).items())
    )
    metadata = {
        "split_counts": split_counts,
        "decision_distribution": decision_distribution,
        "validation_two_pass_count": validation_two_pass,
        "validation_single_pass_count": (
            split_counts["validation"] - validation_two_pass
        ),
        "pilot_test_two_pass_count": sum(
            int(row.get("annotation_pass_count", 0)) >= 2
            for row in split_rows
            if row["dataset_split"] == "pilot_test"
        ),
        "split_seed": seed,
        "freeze_performed": False,
    }
    metadata["pilot_test_single_pass_count"] = (
        metadata["split_counts"]["pilot_test"]
        - metadata["pilot_test_two_pass_count"]
    )
    return split_rows, metadata


def _distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["expected_decision"] for row in rows).items()))


def _split_values(
    rows: list[dict[str, Any]],
    key_builder,
) -> dict[Any, set[str]]:
    values: defaultdict[Any, set[str]] = defaultdict(set)
    for row in rows:
        key = key_builder(row)
        if key not in (None, "", ("", 0)):
            values[key].add(_text(row.get("dataset_split")))
    return values


def audit_selection_and_split(
    *,
    selectable_pool: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    quarantine_rows: list[dict[str, str]],
    dev50_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """审计选择与分层草案；任何漂移都直接失败关闭。"""
    if len(selectable_pool) != int(config["expected_selectable_pool_count"]):
        raise ValueError("selectable pool count drift")
    if len(split_rows) != int(config["expected_selected_count"]):
        raise ValueError("selected count drift")
    if len(quarantine_rows) != int(config["expected_quarantine_count"]):
        raise ValueError("quarantine count drift")

    pool_by_id = {
        _text(row.get("candidate_id")): row for row in selectable_pool
    }
    selected_ids = [_text(row.get("candidate_id")) for row in split_rows]
    if len(set(selected_ids)) != len(selected_ids) or "" in selected_ids:
        raise ValueError("selected candidate identity is not unique")
    if not set(selected_ids).issubset(pool_by_id):
        raise ValueError("selected candidate is absent from selectable pool")
    quarantined_ids = {
        _text(row.get("candidate_id")) for row in quarantine_rows
    }
    if set(selected_ids) & quarantined_ids:
        raise ValueError("quarantined candidate leaked into selection")

    immutable_fields = (
        "question",
        "expected_decision",
        "current_kb_support",
        "gold_evidence_status",
        "required_evidence_type",
        "required_claims",
        "allowed_claims",
        "forbidden_claims",
        "missing_evidence_type",
        "missing_information",
        "risk_labels",
        "source_id",
        "page_number",
        "fact_cluster_id",
        "evidence_anchor_group_id",
        "independence_unit_id",
        "annotation_pass_count",
        "requires_second_pass",
    )
    for row in split_rows:
        parent = pool_by_id[row["candidate_id"]]
        drift = [field for field in immutable_fields if row.get(field) != parent.get(field)]
        if drift:
            raise ValueError(
                f"selection label drift for {row['candidate_id']}: {', '.join(drift)}"
            )
        if row.get("freeze_status") != "draft" or row.get("split_status") != "proposal":
            raise ValueError("selection unexpectedly claims freeze or final split")
        if row.get("gold_status") not in (None, ""):
            raise ValueError("selection unexpectedly claims Gold status")

    expected_total = {
        key: int(value)
        for key, value in config["target_decision_distribution"].items()
        if int(value) > 0
    }
    if _distribution(split_rows) != dict(sorted(expected_total.items())):
        raise ValueError("selected decision distribution drift")
    split_counts = Counter(_text(row.get("dataset_split")) for row in split_rows)
    expected_split_counts = {
        split_name: sum(int(value) for value in target.values())
        for split_name, target in config["split_targets"].items()
    }
    if dict(sorted(split_counts.items())) != dict(sorted(expected_split_counts.items())):
        raise ValueError("split count drift")
    for split_name, target in config["split_targets"].items():
        actual = _distribution(
            [row for row in split_rows if row["dataset_split"] == split_name]
        )
        expected = {
            key: int(value) for key, value in target.items() if int(value) > 0
        }
        if actual != dict(sorted(expected.items())):
            raise ValueError(f"{split_name} decision distribution drift")

    two_pass_count = sum(
        int(row.get("annotation_pass_count", 0)) >= 2 for row in split_rows
    )
    if two_pass_count != int(config["expected_selected_two_pass_count"]):
        raise ValueError("selected two-pass count drift")
    if len(split_rows) - two_pass_count != int(
        config["expected_selected_single_pass_count"]
    ):
        raise ValueError("selected single-pass count drift")
    expected_split_passes = config.get("expected_split_annotation_pass_distribution")
    actual_split_passes: dict[str, dict[str, int]] = {}
    for split_name in sorted(split_counts):
        rows_for_split = [
            row for row in split_rows if row["dataset_split"] == split_name
        ]
        split_two_pass = sum(
            int(row.get("annotation_pass_count", 0)) >= 2 for row in rows_for_split
        )
        actual_split_passes[split_name] = {
            "two_pass": split_two_pass,
            "single_pass": len(rows_for_split) - split_two_pass,
        }
    if expected_split_passes is not None and actual_split_passes != {
        split_name: {
            "two_pass": int(values["two_pass"]),
            "single_pass": int(values["single_pass"]),
        }
        for split_name, values in sorted(expected_split_passes.items())
    }:
        raise ValueError("split annotation-pass distribution drift")

    structural_maps = {
        "fact_cluster": _split_values(
            split_rows, lambda row: _text(row.get("fact_cluster_id"))
        ),
        "anchor_group": _split_values(
            split_rows, lambda row: _text(row.get("evidence_anchor_group_id"))
        ),
        "independence_unit": _split_values(
            split_rows, lambda row: _text(row.get("independence_unit_id"))
        ),
        "source_page": _split_values(
            split_rows,
            lambda row: (
                _text(row.get("source_id")),
                int(row.get("page_number", 0)),
            ),
        ),
    }
    leakage = {
        kind: sorted(str(key) for key, splits in values.items() if len(splits) > 1)
        for kind, values in structural_maps.items()
    }
    leakage = {kind: values for kind, values in leakage.items() if values}
    if leakage:
        raise ValueError(f"cross-split evidence leakage: {leakage}")

    ngram_size = int(config["ngram_size"])
    threshold = float(config["jaccard_threshold"])
    validation_rows = [
        row for row in split_rows if row["dataset_split"] == "validation"
    ]
    pilot_rows = [
        row for row in split_rows if row["dataset_split"] == "pilot_test"
    ]
    cross_split_near_duplicates: list[dict[str, Any]] = []
    for validation in validation_rows:
        left = _question_ngrams(validation["question"], ngram_size)
        for pilot in pilot_rows:
            similarity = _jaccard(
                left,
                _question_ngrams(pilot["question"], ngram_size),
            )
            if similarity >= threshold:
                cross_split_near_duplicates.append(
                    {
                        "validation_id": validation["candidate_id"],
                        "pilot_test_id": pilot["candidate_id"],
                        "similarity": round(similarity, 6),
                    }
                )
    if cross_split_near_duplicates:
        raise ValueError("cross-split near-duplicate question leakage")

    dev_candidate_ids = {
        _text(row.get("sample_id", row.get("candidate_id"))) for row in dev50_rows
    }
    dev_fact_clusters = {
        _text(row.get("fact_cluster_id")) for row in dev50_rows
    }
    dev_anchor_ids = {
        _text(anchor_id)
        for row in dev50_rows
        for anchor_id in row.get("evidence_anchor_ids") or []
    }
    dev_source_pages = {
        (_text(evidence.get("source_id")), int(evidence.get("page")))
        for row in dev50_rows
        for evidence in row.get("gold_evidence") or []
        if evidence.get("source_id") and evidence.get("page") not in (None, "")
    }
    dev_overlap = {
        "candidate_ids": sorted(set(selected_ids) & dev_candidate_ids),
        "fact_clusters": sorted(
            {
                _text(row.get("fact_cluster_id")) for row in split_rows
            }
            & dev_fact_clusters
            - {""}
        ),
        "anchor_ids": sorted(
            {
                _text(anchor_id)
                for row in split_rows
                for anchor_id in row.get("evidence_anchor_ids") or []
            }
            & dev_anchor_ids
            - {""}
        ),
        "source_pages": sorted(
            {
                (_text(row.get("source_id")), int(row.get("page_number")))
                for row in split_rows
            }
            & dev_source_pages
        ),
    }
    if any(dev_overlap.values()):
        raise ValueError(f"development evidence leakage: {dev_overlap}")

    quarantine_distribution = dict(
        sorted(Counter(row["reason"] for row in quarantine_rows).items())
    )
    expected_quarantine = config.get("expected_quarantine_distribution")
    if expected_quarantine is not None and quarantine_distribution != dict(
        sorted((key, int(value)) for key, value in expected_quarantine.items())
    ):
        raise ValueError("quarantine distribution drift")

    return {
        "status": "draft_selection_ready_for_second_pass",
        "selectable_pool_count": len(selectable_pool),
        "selected_count": len(split_rows),
        "decision_distribution": _distribution(split_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "split_decision_distribution": {
            split_name: _distribution(
                [row for row in split_rows if row["dataset_split"] == split_name]
            )
            for split_name in sorted(split_counts)
        },
        "selected_two_pass_count": two_pass_count,
        "selected_single_pass_count": len(split_rows) - two_pass_count,
        "split_annotation_pass_distribution": actual_split_passes,
        "quarantine_count": len(quarantine_rows),
        "quarantine_distribution": quarantine_distribution,
        "cross_split_structural_overlap_count": 0,
        "cross_split_near_duplicate_count": 0,
        "development_overlap_count": 0,
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
    }


def _summary_markdown(
    *,
    config: dict[str, Any],
    audit: dict[str, Any],
    output_hashes: dict[str, str],
) -> str:
    decisions = audit["decision_distribution"]
    split_counts = audit["split_counts"]
    quarantine = audit["quarantine_distribution"]
    lines = [
        "# Benchmark120 选择与分层草案 v0.1",
        "",
        "## 结论",
        "",
        "- 当前产物是选择与分层草案，不是 Gold Benchmark，也未冻结。",
        f"- 可选池：{audit['selectable_pool_count']} 条；选中：{audit['selected_count']} 条。",
        f"- Validation：{split_counts['validation']} 条；Pilot Test：{split_counts['pilot_test']} 条。",
        (
            "- 决策分布："
            f"answer={decisions.get('answer', 0)}，"
            f"review_required={decisions.get('review_required', 0)}，"
            f"insufficient_evidence={decisions.get('insufficient_evidence', 0)}，"
            f"boundary_refusal={decisions.get('boundary_refusal', 0)}。"
        ),
        (
            "- 标注质量："
            f"两轮一致={audit['selected_two_pass_count']}，"
            f"单轮作者终审且待第二轮={audit['selected_single_pass_count']}。"
        ),
        (
            "- 隔离记录："
            f"损坏裁决理由={quarantine.get('corrupt_resolution_reason', 0)}，"
            f"两轮不一致={quarantine.get('pass1_pass2_disagreement', 0)}，"
            f"第二轮未接受={quarantine.get('pass2_not_accepted', 0)}。"
        ),
        "- 跨分层事实、证据锚点、独立单元、来源页及近重复问题泄漏：0。",
        "- 外部模型调用：0；input_tokens=0；output_tokens=0；estimated_cost=0。",
        "",
        "## 下一步",
        "",
        "1. 对选中的单轮作者终审样本执行独立第二轮盲审。",
        "2. 第二轮完成后再运行 Gold promotion gate。",
        "3. 通过 promotion gate 后才允许冻结 Validation/Pilot Test。",
        "",
        "## 版本",
        "",
        f"- dataset_version: `{config['dataset_version']}`",
        f"- kb_version: `{config['kb_version']}`",
        f"- selection_seed: `{config['selection_seed']}`",
        "",
        "## 产物哈希",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{digest}`" for name, digest in sorted(output_hashes.items())
    )
    return "\n".join(lines) + "\n"


def run_selection(
    *,
    config_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    config = _load_json(config_path)
    if config.get("gold_promotion_allowed") is not False:
        raise ValueError("B3.6i must not allow Gold promotion")
    if config.get("freeze_allowed") is not False:
        raise ValueError("B3.6i must not allow dataset freezing")
    input_hashes = verify_input_hashes(config["input_sha256"], base_dir=root)
    paths = {
        key: root / value for key, value in config["input_paths"].items()
    }

    pass2_rows = _load_csv(paths["pass2_queue"])
    linkage_payload = _load_json(paths["pass2_linkage"])
    linkage_rows = linkage_payload.get("records")
    if not isinstance(linkage_rows, list):
        raise ValueError("pass2 linkage records are missing")
    resolution_rows = _load_csv(paths["resolution"])
    new_reviewed_rows = _load_jsonl(paths["new_reviewed_pool"])
    dev50_rows = _load_jsonl(paths["dev50"])

    selectable_pool, quarantine_rows = build_selectable_pool(
        pass2_rows=pass2_rows,
        linkage_rows=linkage_rows,
        resolution_rows=resolution_rows,
        new_reviewed_rows=new_reviewed_rows,
        config=config,
    )
    selectable_pool = [
        {
            **row,
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "selection_version": config["selection_version"],
        }
        for row in selectable_pool
    ]
    selected_rows, selection_metadata = select_benchmark120(
        selectable_pool,
        target_distribution=config["target_decision_distribution"],
        seed=int(config["selection_seed"]),
        ngram_size=int(config["ngram_size"]),
        similarity_threshold=float(config["jaccard_threshold"]),
    )
    split_rows, split_metadata = propose_grouped_split(
        selected_rows,
        split_targets=config["split_targets"],
        desired_validation_two_pass=int(
            config["desired_validation_two_pass_count"]
        ),
        seed=int(config["selection_seed"]),
        ngram_size=int(config["ngram_size"]),
        similarity_threshold=float(config["jaccard_threshold"]),
    )
    audit = audit_selection_and_split(
        selectable_pool=selectable_pool,
        split_rows=split_rows,
        quarantine_rows=quarantine_rows,
        dev50_rows=dev50_rows,
        config=config,
    )
    audit.update(
        {
            "config_version": config["config_version"],
            "selection_version": config["selection_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "input_sha256": input_hashes,
            "selection_metadata": selection_metadata,
            "split_metadata": split_metadata,
            "quarantine_records": quarantine_rows,
        }
    )

    output_dir = root / config["output_dir"]
    outputs = {
        key: output_dir / filename for key, filename in config["outputs"].items()
    }
    split_proposal = {
        "status": audit["status"],
        "selection_version": config["selection_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "selection_metadata": selection_metadata,
        "split_metadata": split_metadata,
        "validation_candidate_ids": [
            row["candidate_id"]
            for row in split_rows
            if row["dataset_split"] == "validation"
        ],
        "pilot_test_candidate_ids": [
            row["candidate_id"]
            for row in split_rows
            if row["dataset_split"] == "pilot_test"
        ],
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "usage": audit["usage"],
    }

    _write_jsonl(outputs["selectable_pool"], selectable_pool)
    _write_jsonl(outputs["selection_draft"], split_rows)
    _write_json(outputs["split_proposal"], split_proposal)
    output_hashes = {
        key: file_sha256(outputs[key])
        for key in ("selectable_pool", "selection_draft", "split_proposal")
    }
    audit["output_sha256"] = output_hashes
    _write_json(outputs["audit"], audit)
    summary = _summary_markdown(
        config=config,
        audit=audit,
        output_hashes={
            **output_hashes,
            "audit": file_sha256(outputs["audit"]),
        },
    )
    _atomic_write_text(outputs["summary"], summary)
    return {
        "status": audit["status"],
        "selected_count": audit["selected_count"],
        "split_counts": audit["split_counts"],
        "selected_two_pass_count": audit["selected_two_pass_count"],
        "selected_single_pass_count": audit["selected_single_pass_count"],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": {
            **output_hashes,
            "audit": file_sha256(outputs["audit"]),
            "summary": file_sha256(outputs["summary"]),
        },
        "usage": audit["usage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit the Benchmark120 draft selection."
    )
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--config",
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark120_selection_v0_1.json"
        ),
    )
    parser.add_argument("--repo-root", default=default_root)
    args = parser.parse_args()
    result = run_selection(config_path=args.config, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _old_selectable_row(
    pass2: dict[str, Any],
    linkage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": _text(linkage.get("candidate_id")),
        "question": _text(pass2.get("pass2_final_question")),
        "expected_decision": _text(pass2.get("pass2_expected_decision")),
        "current_kb_support": _text(pass2.get("pass2_current_kb_support")),
        "gold_evidence_status": _text(pass2.get("pass2_gold_evidence_status")),
        "required_evidence_type": _json_list(
            pass2.get("pass2_required_evidence_type")
        ),
        "required_claims": _json_list(pass2.get("pass2_required_claims")),
        "allowed_claims": _json_list(pass2.get("pass2_allowed_claims")),
        "forbidden_claims": _json_list(pass2.get("pass2_forbidden_claims")),
        "missing_evidence_type": _json_list(
            pass2.get("pass2_missing_evidence_type")
        ),
        "missing_information": _json_list(
            pass2.get("pass2_missing_information")
        ),
        "risk_labels": _json_list(pass2.get("pass2_risk_labels")),
        "source_id": _text(pass2.get("source_id")),
        "source_title": _text(pass2.get("source_title")),
        "source_filename": _text(pass2.get("source_filename")),
        "source_sha256": _text(pass2.get("source_sha256")),
        "source_type": _text(pass2.get("source_type")),
        "source_year": _text(pass2.get("source_year")),
        "jurisdiction": _text(pass2.get("jurisdiction")),
        "page_number": int(pass2.get("page_number")),
        "anchor_text_span": _text(pass2.get("anchor_text_span")),
        "evidence_scope": _text(pass2.get("evidence_scope")),
        "age_scope": _text(pass2.get("age_scope")),
        "applicability_conditions": _text(
            pass2.get("applicability_conditions")
        ),
        "supported_claim_types": _json_list(
            pass2.get("supported_claim_types")
        ),
        "policy_rule_ids": _json_list(pass2.get("policy_rule_ids")),
        "evidence_anchor_ids": _json_list(linkage.get("evidence_anchor_ids")),
        "evidence_anchor_group_id": _text(
            linkage.get("evidence_anchor_group_id")
        ),
        "fact_cluster_id": _text(linkage.get("provisional_fact_cluster_id")),
        "independence_unit_id": _text(linkage.get("independence_unit_id")),
        "origin_pool": _text(linkage.get("origin_pool")),
        "source_row_sha256": _text(linkage.get("source_row_sha256")),
        "annotation_pass_count": 2,
        "requires_second_pass": False,
        "candidate_status": "selectable_two_pass_consistent",
        "freeze_status": "draft",
    }


def _new_selectable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _text(row.get("candidate_id")),
        "question": _text(row.get("question")),
        "expected_decision": _text(row.get("reviewed_expected_decision")),
        "current_kb_support": _text(row.get("current_kb_support")),
        "gold_evidence_status": _text(
            row.get("gold_evidence_status", "page_span_located")
        ),
        "required_evidence_type": list(row.get("required_evidence_type") or []),
        "required_claims": list(row.get("required_claims") or []),
        "allowed_claims": [_text(row.get("reviewed_allowed_answer_scope"))],
        "forbidden_claims": list(row.get("reviewed_forbidden_claims") or []),
        "missing_evidence_type": list(row.get("missing_evidence_type") or []),
        "missing_information": list(row.get("missing_information") or []),
        "risk_labels": list(row.get("reviewed_risk_labels") or []),
        "source_id": _text(row.get("source_id")),
        "source_title": _text(row.get("source_title")),
        "source_filename": _text(row.get("source_filename")),
        "source_sha256": _text(row.get("source_sha256")),
        "source_type": _text(row.get("source_type")),
        "source_year": _text(row.get("source_year")),
        "jurisdiction": _text(row.get("jurisdiction")),
        "page_number": int(row.get("page_number")),
        "anchor_text_span": _text(row.get("anchor_text_span")),
        "evidence_scope": _text(row.get("evidence_scope")),
        "age_scope": _text(row.get("age_scope")),
        "applicability_conditions": _text(row.get("applicability_conditions")),
        "supported_claim_types": list(row.get("supported_claim_types") or []),
        "policy_rule_ids": list(row.get("policy_rule_ids") or []),
        "evidence_anchor_ids": list(row.get("evidence_anchor_ids") or []),
        "evidence_anchor_group_id": _text(row.get("evidence_anchor_group_id")),
        "fact_cluster_id": _text(row.get("provisional_fact_cluster_id")),
        "independence_unit_id": _text(row.get("independence_unit_id")),
        "origin_pool": "anchor_supplement_reviewed",
        "source_row_sha256": _text(row.get("source_row_sha256")),
        "annotation_pass_count": 1,
        "requires_second_pass": True,
        "candidate_status": "selectable_author_reviewed_pending_second_pass",
        "freeze_status": "draft",
    }


def build_selectable_pool(
    *,
    pass2_rows: list[dict[str, Any]],
    linkage_rows: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    new_reviewed_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """构建可进入 Benchmark120 选择阶段的保守候选池。"""
    pass2_by_id = {
        _text(row.get("pass2_item_id")): row for row in pass2_rows
    }
    resolution_ids = {
        _text(row.get("candidate_id")) for row in resolution_rows
    }
    corrupt_pattern = re.compile(config.get("corrupt_reason_pattern", r"^\?+$"))
    pool: list[dict[str, Any]] = []
    quarantine: list[dict[str, str]] = []

    for linkage in sorted(
        linkage_rows,
        key=lambda row: _text(row.get("candidate_id")),
    ):
        candidate_id = _text(linkage.get("candidate_id"))
        pass2 = pass2_by_id.get(_text(linkage.get("pass2_item_id")))
        if pass2 is None:
            quarantine.append(
                {"candidate_id": candidate_id, "reason": "missing_pass2_record"}
            )
            continue
        if _text(pass2.get("pass2_outcome")) != "accepted":
            quarantine.append(
                {"candidate_id": candidate_id, "reason": "pass2_not_accepted"}
            )
            continue
        if candidate_id in resolution_ids:
            resolution = next(
                row
                for row in resolution_rows
                if _text(row.get("candidate_id")) == candidate_id
            )
            reason = _text(resolution.get("resolution_reason"))
            quarantine.append(
                {
                    "candidate_id": candidate_id,
                    "reason": (
                        "corrupt_resolution_reason"
                        if corrupt_pattern.search(reason)
                        else "author_resolution_excluded"
                    ),
                }
            )
            continue

        comparable = (
            ("outcome", "pass1_outcome", "pass2_outcome"),
            ("question", "pass1_final_question", "pass2_final_question"),
            (
                "decision",
                "pass1_expected_decision",
                "pass2_expected_decision",
            ),
            (
                "kb_support",
                "pass1_current_kb_support",
                "pass2_current_kb_support",
            ),
            (
                "gold_evidence",
                "pass1_gold_evidence_status",
                "pass2_gold_evidence_status",
            ),
        )
        disagreements = [
            label
            for label, pass1_field, pass2_field in comparable
            if _text(linkage.get(pass1_field)) != _text(pass2.get(pass2_field))
        ]
        if disagreements:
            quarantine.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "pass1_pass2_disagreement",
                }
            )
            continue
        pool.append(_old_selectable_row(pass2, linkage))

    for row in sorted(
        new_reviewed_rows,
        key=lambda item: _text(item.get("candidate_id")),
    ):
        candidate_id = _text(row.get("candidate_id"))
        if (
            _text(row.get("author_outcome")) != "accepted"
            or _text(row.get("candidate_status")) != "author_reviewed_candidate"
        ):
            quarantine.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "new_candidate_not_author_accepted",
                }
            )
            continue
        pool.append(_new_selectable_row(row))

    return sorted(pool, key=lambda row: row["candidate_id"]), quarantine


if __name__ == "__main__":
    main()
