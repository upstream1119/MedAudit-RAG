import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_overlap_audit.py"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "benchmark_v1"
    / "benchmark_candidates_v0_1.jsonl"
)
DEV50_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "dev50"
    / "dev50_v1_0_frozen.jsonl"
)
REGISTRY_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "dev50"
    / "evidence_anchor_registry.md"
)
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_overlap_audit_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark overlap audit is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_overlap_audit",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(**overrides) -> dict:
    row = {
        "candidate_id": "PMSQA-BV1C-test-001",
        "question": "儿童肺炎治疗后是否需要再次评估？",
        "source_id": "SRC-101",
        "page_number": 8,
        "evidence_anchor_ids": ["ANCH-new-001"],
        "provisional_fact_cluster_id": "FC-new-001",
        "evidence_anchor_group_id": "EAG-new-001",
        "candidate_status": "draft_candidate_unverified",
    }
    row.update(overrides)
    return row


def _dev50(**overrides) -> dict:
    row = {
        "sample_id": "PMSQA_DEV_001",
        "question": "儿童肺炎治疗后是否需要再次评估？",
        "evidence_anchor_ids": ["ANCH-dev-001"],
        "fact_cluster_id": "FC-dev-001",
    }
    row.update(overrides)
    return row


def _pass1_row(**overrides) -> dict:
    row = {
        "candidate_id": "PMSQA-BV1C-test-001",
        "question": "原始问题",
        "pass1_outcome": "revise",
        "pass1_final_question": "修订后的儿科用药证据边界问题",
        "independence_unit_id": "IU-test-001",
    }
    row.update(overrides)
    return row


def test_exact_dev50_question_is_rejected():
    module = _load_module()

    result = module.audit_candidate_overlap(
        [_candidate()],
        [_dev50()],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    audited = result["audited_candidates"][0]
    assert audited["dev50_overlap_status"] == "rejected"
    assert audited["overlap_decision"] == "reject"
    assert "exact_question" in audited["overlap_reasons"]


def test_reused_dev50_anchor_or_fact_cluster_is_rejected():
    module = _load_module()
    candidates = [
        _candidate(
            candidate_id="PMSQA-BV1C-anchor",
            question="完全不同的锚点复用问题",
            evidence_anchor_ids=["ANCH-dev-001"],
        ),
        _candidate(
            candidate_id="PMSQA-BV1C-fact",
            question="完全不同的事实簇复用问题",
            provisional_fact_cluster_id="FC-dev-001",
        ),
    ]

    result = module.audit_candidate_overlap(
        candidates,
        [_dev50()],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    by_id = {row["candidate_id"]: row for row in result["audited_candidates"]}
    assert "evidence_anchor_id" in by_id["PMSQA-BV1C-anchor"]["overlap_reasons"]
    assert "fact_cluster_id" in by_id["PMSQA-BV1C-fact"]["overlap_reasons"]
    assert all(row["overlap_decision"] == "reject" for row in by_id.values())


def test_reused_dev50_source_page_is_rejected_as_anchor_leakage():
    module = _load_module()

    result = module.audit_candidate_overlap(
        [_candidate(question="不同表述但仍来自同一证据页")],
        [_dev50(question="另一道开发问题")],
        {("SRC-101", 8): ["ANCH-dev-page"]},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    audited = result["audited_candidates"][0]
    assert audited["overlap_decision"] == "reject"
    assert "source_page_anchor" in audited["overlap_reasons"]
    assert audited["dev50_overlap_anchor_ids"] == ["ANCH-dev-page"]


def test_chinese_character_trigram_jaccard_is_deterministic():
    module = _load_module()

    assert module.jaccard_similarity("儿童肺炎复评", "儿童肺炎复评", 3) == 1.0
    assert module.jaccard_similarity("儿童肺炎复评", "肾功能监测", 3) == 0.0
    assert module.jaccard_similarity(
        "儿童肺炎复评",
        "儿童肺炎评估",
        3,
    ) == pytest.approx(1 / 3)


def test_revised_question_reaudit_excludes_self_and_same_independence_unit():
    module = _load_module()
    target = _pass1_row()
    same_unit = _pass1_row(
        candidate_id="PMSQA-BV1C-test-002",
        question="同一证据锚点的角色变体",
        pass1_outcome="accepted",
        pass1_final_question=target["pass1_final_question"],
    )
    unrelated = _pass1_row(
        candidate_id="PMSQA-BV1C-test-003",
        question="完全无关的问题",
        pass1_outcome="",
        pass1_final_question="",
        independence_unit_id="IU-test-002",
    )

    report = module.audit_revised_question_overlap(
        target,
        [target, same_unit, unrelated],
        [_dev50(question="另一道开发集问题")],
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    assert report["reaudit_decision"] == "clear"
    assert report["internal_overlap_status"] == "clear"
    assert report["compared_candidate_count"] == 1


def test_revised_question_reaudit_rejects_exact_dev50_question():
    module = _load_module()
    target = _pass1_row(pass1_final_question="儿童肺炎治疗后是否需要再次评估？")

    report = module.audit_revised_question_overlap(
        target,
        [target],
        [_dev50(question=target["pass1_final_question"])],
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    assert report["reaudit_decision"] == "reject"
    assert report["dev50_overlap_status"] == "rejected"
    assert "exact_question_dev50" in report["reaudit_reasons"]


def test_revised_question_reaudit_flags_cross_group_near_duplicate():
    module = _load_module()
    target = _pass1_row(
        pass1_final_question="儿童肺炎治疗以后是否需要再次评估？",
    )
    cross_group = _pass1_row(
        candidate_id="PMSQA-BV1C-test-002",
        question="儿童肺炎治疗后是否需要再次评估？",
        pass1_outcome="accepted",
        pass1_final_question="儿童肺炎治疗后是否需要再次评估？",
        independence_unit_id="IU-test-002",
    )

    report = module.audit_revised_question_overlap(
        target,
        [target, cross_group],
        [_dev50(question="另一道开发集问题")],
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    assert report["reaudit_decision"] == "needs_review"
    assert report["internal_overlap_status"] == "needs_review"
    assert report["max_internal_similar_candidate_id"] == cross_group["candidate_id"]


def test_run_revised_question_reaudit_writes_parent_hashes_and_zero_usage(tmp_path):
    module = _load_module()
    queue_path = tmp_path / "pass1.csv"
    dev50_path = tmp_path / "dev50.jsonl"
    parent_audit_path = tmp_path / "overlap_audit.json"
    output_path = tmp_path / "reaudit.json"
    target = _pass1_row()
    with queue_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(target))
        writer.writeheader()
        writer.writerow(target)
    dev50_path.write_text(
        json.dumps(_dev50(question="另一道开发集问题"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    parent_audit_path.write_text("{}\n", encoding="utf-8")

    report = module.run_revised_question_reaudit(
        candidate_id=target["candidate_id"],
        pass1_queue_path=queue_path,
        dev50_path=dev50_path,
        config_path=CONFIG_PATH,
        parent_audit_path=parent_audit_path,
        output_path=output_path,
    )

    assert report["reaudit_decision"] == "clear"
    assert report["parent_artifacts"]["queue_sha256"] == module._compute_sha256(
        queue_path
    )
    assert report["parent_artifacts"]["overlap_audit_sha256"] == module._compute_sha256(
        parent_audit_path
    )
    assert report["usage"]["external_model_calls"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["candidate_id"] == target[
        "candidate_id"
    ]


def test_similarity_at_threshold_enters_manual_review_queue():
    module = _load_module()
    candidate = _candidate(
        question="儿童肺炎治疗以后是否需要再次评估？",
    )

    result = module.audit_candidate_overlap(
        [candidate],
        [_dev50(question="儿童肺炎治疗后是否需要再次评估？")],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    audited = result["audited_candidates"][0]
    assert audited["dev50_overlap_status"] == "needs_review"
    assert audited["overlap_decision"] == "needs_review"
    assert audited["max_dev50_question_similarity"] >= 0.65
    assert result["review_queue"][0]["comparison_scope"] == "dev50"
    assert result["review_queue"][0]["manual_decision"] == ""


def test_same_anchor_variants_share_one_independence_unit():
    module = _load_module()
    candidates = [
        _candidate(
            candidate_id="PMSQA-BV1C-direct",
            question="儿童用药监测有哪些直接证据？",
        ),
        _candidate(
            candidate_id="PMSQA-BV1C-boundary",
            question="该监测证据能否支持个体化处方？",
        ),
    ]

    result = module.audit_candidate_overlap(
        candidates,
        [_dev50(question="完全无关的开发集问题")],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    audited = result["audited_candidates"]
    assert {row["internal_overlap_status"] for row in audited} == {
        "group_linked"
    }
    assert len({row["independence_unit_id"] for row in audited}) == 1
    assert all(
        row["same_group_candidate_ids"]
        == ["PMSQA-BV1C-boundary", "PMSQA-BV1C-direct"]
        for row in audited
    )
    assert all(row["overlap_decision"] == "keep" for row in audited)


def test_internal_exact_duplicate_rejects_noncanonical_candidate():
    module = _load_module()
    candidates = [
        _candidate(
            candidate_id="PMSQA-BV1C-b",
            provisional_fact_cluster_id="FC-b",
            evidence_anchor_group_id="EAG-b",
        ),
        _candidate(
            candidate_id="PMSQA-BV1C-a",
            provisional_fact_cluster_id="FC-a",
            evidence_anchor_group_id="EAG-a",
        ),
    ]

    result = module.audit_candidate_overlap(
        candidates,
        [_dev50(question="无关开发问题")],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    by_id = {row["candidate_id"]: row for row in result["audited_candidates"]}
    assert by_id["PMSQA-BV1C-a"]["overlap_decision"] == "keep"
    assert by_id["PMSQA-BV1C-b"]["overlap_decision"] == "reject"
    assert "exact_question_internal" in by_id["PMSQA-BV1C-b"]["overlap_reasons"]
    assert by_id["PMSQA-BV1C-b"]["internal_overlap_status"] == "rejected"


def test_internal_near_duplicate_across_groups_enters_review_queue():
    module = _load_module()
    candidates = [
        _candidate(
            candidate_id="PMSQA-BV1C-near-a",
            question="儿童肺炎治疗后是否需要再次评估？",
            provisional_fact_cluster_id="FC-near-a",
            evidence_anchor_group_id="EAG-near-a",
        ),
        _candidate(
            candidate_id="PMSQA-BV1C-near-b",
            question="儿童肺炎治疗以后是否需要再次评估？",
            provisional_fact_cluster_id="FC-near-b",
            evidence_anchor_group_id="EAG-near-b",
        ),
    ]

    result = module.audit_candidate_overlap(
        candidates,
        [_dev50(question="无关开发问题")],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    audited = result["audited_candidates"]
    assert all(row["overlap_decision"] == "needs_review" for row in audited)
    assert all(row["internal_overlap_status"] == "needs_review" for row in audited)
    internal_queue = [
        row
        for row in result["review_queue"]
        if row["comparison_scope"] == "candidate_internal"
    ]
    assert len(internal_queue) == 1
    assert internal_queue[0]["similarity"] >= 0.65


def test_unresolved_review_blocks_deduplicated_output():
    module = _load_module()
    result = module.audit_candidate_overlap(
        [_candidate(question="儿童肺炎治疗以后是否需要再次评估？")],
        [_dev50(question="儿童肺炎治疗后是否需要再次评估？")],
        {},
        {"ngram_size": 3, "jaccard_threshold": 0.65},
    )

    with pytest.raises(ValueError, match="人工复核"):
        module.select_deduplicated_candidates(
            result,
            {
                "fail_on_unresolved_review": True,
                "audited_candidate_status": "overlap_audited_draft",
                "audit_version": "test-audit-v0.1",
            },
        )


def test_real_candidate_pool_writes_complete_deterministic_audit(tmp_path):
    module = _load_module()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = module.run_overlap_audit(
        candidates_path=CANDIDATE_PATH,
        dev50_path=DEV50_PATH,
        registry_path=REGISTRY_PATH,
        config_path=CONFIG_PATH,
        output_dir=first_dir,
    )
    second = module.run_overlap_audit(
        candidates_path=CANDIDATE_PATH,
        dev50_path=DEV50_PATH,
        registry_path=REGISTRY_PATH,
        config_path=CONFIG_PATH,
        output_dir=second_dir,
    )

    assert first["counts"] == second["counts"]
    assert first["counts"] == {
        "input_candidates": 144,
        "kept_candidates": 144,
        "rejected_candidates": 0,
        "review_queue": 0,
        "unresolved_review": 0,
        "independence_units": 72,
    }
    assert first["similarity"]["ngram_size"] == 3
    assert first["similarity"]["jaccard_threshold"] == 0.65
    assert first["grouping"]["group_size_distribution"] == {"2": 72}
    assert first["manual_review"]["conclusions"] == []
    assert first["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    output_names = {
        "overlap_audit_v0_1.json",
        "overlap_review_queue_v0_1.csv",
        "benchmark_candidates_v0_2_deduplicated.jsonl",
    }
    assert {path.name for path in first_dir.iterdir()} == output_names
    for name in output_names:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
