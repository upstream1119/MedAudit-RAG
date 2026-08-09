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
    / "benchmark_annotation_validator.py"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "benchmark_v1"
    / "benchmark_candidates_v0_2_deduplicated.jsonl"
)
ANCHOR_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "benchmark_v1"
    / "evidence_anchor_pool_v0_1.jsonl"
)
MANIFEST_PATH = REPO_ROOT / "data" / "guidelines" / "source_manifest.json"
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_annotation_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark annotation validator is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_annotation_validator",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _anchor(**overrides) -> dict:
    row = {
        "anchor_id": "ANCH-test-001",
        "source_id": "SRC-101",
        "source_title": "测试儿科指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": "a" * 64,
        "page_number": 8,
        "text_span": "指南建议治疗后四十八至七十二小时重新评估疗效。",
        "evidence_scope": "儿童肺炎治疗后的疗效复评。",
        "age_scope": "儿童",
        "applicability_conditions": "已开始治疗且需要观察疗效。",
        "supported_claim_types": ["treatment_reassessment"],
        "scope_check": "within_can_support",
        "verification_status": "author_verified_anchor",
    }
    row.update(overrides)
    return row


def _candidate(**overrides) -> dict:
    row = {
        "candidate_id": "PMSQA-BV1C-test-001",
        "question": "儿童肺炎治疗后是否需要再次评估？",
        "candidate_role": "direct_support",
        "candidate_status": "overlap_audited_draft",
        "dataset_version": "benchmark-v1.0-overlap-audited-draft",
        "kb_version": "KB-medium-v1",
        "schema_version": "benchmark-schema-v0.1",
        "protocol_version": "benchmark-protocol-v0.1",
        "source_id": "SRC-101",
        "source_title": "测试儿科指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": "a" * 64,
        "page_number": 8,
        "anchor_text_span": "指南建议治疗后四十八至七十二小时重新评估疗效。",
        "evidence_scope": "儿童肺炎治疗后的疗效复评。",
        "age_scope": "儿童",
        "applicability_conditions": "已开始治疗且需要观察疗效。",
        "supported_claim_types": ["treatment_reassessment"],
        "scope_check": "within_can_support",
        "evidence_anchor_ids": ["ANCH-test-001"],
        "evidence_anchor_group_id": "EAG-test-001",
        "provisional_fact_cluster_id": "FC-test-001",
        "independence_unit_id": "IU-test-001",
        "provisional_expected_decision": "answer",
        "provisional_scenario_type": "evidence-scope",
        "provisional_risk_labels": ["answer", "treatment_reassessment"],
        "current_kb_support": "supported_by_current_kb",
        "missing_evidence_type": [],
        "policy_rule_ids": [],
        "overlap_decision": "keep",
        "dev50_overlap_status": "clear",
        "internal_overlap_status": "group_linked",
    }
    row.update(overrides)
    return row


def _manifest(**overrides) -> dict:
    source = {
        "source_id": "SRC-101",
        "title": "测试儿科指南",
        "filename": "test-guideline.pdf",
        "sha256": "a" * 64,
        "source_type": "clinical_guideline",
        "year": 2024,
        "jurisdiction": "CN",
        "status": "indexed",
        "included_in_kb": True,
        "can_support": ["儿童肺炎治疗后的疗效复评"],
        "cannot_support": ["替代个体化处方"],
    }
    source.update(overrides)
    return {"schema_version": "1.0", "sources": [source]}


def _config(**overrides) -> dict:
    row = {
        "config_version": "benchmark-annotation-config-v0.1",
        "annotation_version": "benchmark-annotation-v0.1",
        "input_dataset_version": "benchmark-v1.0-overlap-audited-draft",
        "output_dataset_version": "benchmark-v1.0-pass1-pending",
        "schema_version": "benchmark-schema-v0.1",
        "protocol_version": "benchmark-protocol-v0.1",
        "kb_version": "KB-medium-v1",
        "expected_candidate_count": 1,
        "expected_independence_unit_count": 1,
        "pass1_shuffle_seed": 20260729,
        "annotator_role": "author",
        "allowed_outcomes": ["accepted", "revise", "reject"],
        "allowed_decisions": [
            "answer",
            "review_required",
            "insufficient_evidence",
            "boundary_refusal",
        ],
        "allowed_kb_support": [
            "supported_by_current_kb",
            "partially_supported_by_current_kb",
            "not_supported_by_current_kb",
        ],
        "allowed_gold_evidence_status": [
            "page_span_located",
            "missing_source",
            "policy_rule",
        ],
        "policy_rule_id": "MSP-BOUNDARY-001",
        "fail_closed": True,
        "external_model_calls": 0,
    }
    row.update(overrides)
    return row


def _completed_pass1(module, **overrides) -> dict:
    row = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )[0]
    row.update(
        {
            "pass1_reviewer_id": "author-pyh",
            "pass1_annotator_role": "author",
            "pass1_reviewed_at": "2026-07-29T20:30:00+08:00",
            "pass1_outcome": "accepted",
            "pass1_final_question": row["question"],
            "pass1_expected_decision": "answer",
            "pass1_current_kb_support": "supported_by_current_kb",
            "pass1_gold_evidence_status": "page_span_located",
            "pass1_required_evidence_type": ["monitoring"],
            "pass1_required_claims": ["治疗后需要重新评估"],
            "pass1_allowed_claims": ["说明复评时间窗口"],
            "pass1_forbidden_claims": ["替代个体化处方"],
            "pass1_missing_information": [],
            "pass1_risk_labels": ["monitoring"],
            "pass1_issues_found": [],
            "pass1_review_reason": "证据页码、片段和问题边界一致。",
        }
    )
    row.update(overrides)
    return row


def _pass1_decision(module, candidate_id="PMSQA-BV1C-test-001", **overrides) -> dict:
    row = _completed_pass1(module, candidate_id=candidate_id)
    decision = {
        field: row[field]
        for field in module.PASS1_REVIEW_FIELDS
    }
    decision["candidate_id"] = candidate_id
    decision.update(overrides)
    return decision


def test_pass1_queue_only_admits_overlap_audited_kept_candidates():
    module = _load_module()
    invalid = _candidate(
        candidate_id="PMSQA-BV1C-rejected",
        overlap_decision="reject",
    )

    with pytest.raises(ValueError, match="overlap_decision"):
        module.build_pass1_queue(
            [_candidate(), invalid],
            [_anchor()],
            _manifest(),
            _config(expected_candidate_count=2),
        )


@pytest.mark.parametrize(
    ("candidate_change", "anchor_change", "manifest_change", "message"),
    [
        ({"source_sha256": "b" * 64}, {}, {}, "SHA-256"),
        ({"page_number": 9}, {}, {}, "页码"),
        ({"anchor_text_span": "错误片段"}, {}, {}, "证据片段"),
        ({}, {}, {"title": "另一份指南"}, "标题"),
    ],
)
def test_provenance_mismatch_fails_closed(
    candidate_change,
    anchor_change,
    manifest_change,
    message,
):
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module.build_pass1_queue(
            [_candidate(**candidate_change)],
            [_anchor(**anchor_change)],
            _manifest(**manifest_change),
            _config(),
        )


def test_pass1_queue_is_deterministic_and_author_fields_start_blank():
    module = _load_module()
    candidates = [
        _candidate(),
        _candidate(
            candidate_id="PMSQA-BV1C-test-002",
            question="该证据能否直接替代个体化处方？",
            candidate_role="scope_boundary",
            provisional_expected_decision="boundary_refusal",
            provisional_risk_labels=["boundary_refusal"],
            policy_rule_ids=["MSP-BOUNDARY-001"],
        ),
    ]
    config = _config(expected_candidate_count=2)

    first = module.build_pass1_queue(candidates, [_anchor()], _manifest(), config)
    second = module.build_pass1_queue(
        list(reversed(candidates)),
        [_anchor()],
        _manifest(),
        config,
    )

    assert [row["candidate_id"] for row in first] == [
        row["candidate_id"] for row in second
    ]
    assert {row["independence_unit_id"] for row in first} == {"IU-test-001"}
    assert [row["annotation_order"] for row in first] == [1, 2]
    assert all(row["pass1_outcome"] == "" for row in first)
    assert all(row["pass1_required_claims"] == [] for row in first)


def test_csv_json_fields_round_trip(tmp_path):
    module = _load_module()
    rows = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )
    output_path = tmp_path / "pass1.csv"

    module.write_pass1_queue(rows, output_path)
    loaded = module.read_pass1_queue(output_path)

    assert loaded == rows
    with output_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        raw = next(csv.DictReader(file_obj))
    assert json.loads(raw["supported_claim_types"]) == [
        "treatment_reassessment"
    ]


def test_incomplete_pass1_cannot_be_validated():
    module = _load_module()
    pending = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )

    with pytest.raises(ValueError, match="首轮核验未完成"):
        module.validate_completed_pass1(
            pending,
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_annotator_role_must_be_author():
    module = _load_module()
    row = _completed_pass1(module, pass1_annotator_role="clinical_expert")

    with pytest.raises(ValueError, match="author"):
        module.validate_completed_pass1(
            [row],
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_page_span_located_requires_traceable_gold_evidence():
    module = _load_module()
    row = _completed_pass1(module, anchor_text_span="")

    with pytest.raises(ValueError, match="gold evidence"):
        module.validate_completed_pass1(
            [row],
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_missing_source_requires_missing_evidence_type():
    module = _load_module()
    row = _completed_pass1(
        module,
        pass1_expected_decision="insufficient_evidence",
        pass1_current_kb_support="not_supported_by_current_kb",
        pass1_gold_evidence_status="missing_source",
        pass1_missing_information=[],
    )

    with pytest.raises(ValueError, match="missing evidence type"):
        module.validate_completed_pass1(
            [row],
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_boundary_refusal_requires_policy_rule_and_forbidden_claims():
    module = _load_module()
    candidate = _candidate(
        provisional_expected_decision="boundary_refusal",
        policy_rule_ids=[],
    )
    row = _completed_pass1(
        module,
        pass1_expected_decision="boundary_refusal",
        pass1_gold_evidence_status="policy_rule",
        pass1_forbidden_claims=[],
        policy_rule_ids=[],
    )

    with pytest.raises(ValueError, match="policy rule.*forbidden claims"):
        module.validate_completed_pass1(
            [row],
            [candidate],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_revised_question_requires_overlap_reaudit():
    module = _load_module()
    row = _completed_pass1(
        module,
        pass1_outcome="revise",
        pass1_final_question="儿童肺炎治疗四十八小时后应如何复评？",
    )

    summary = module.validate_completed_pass1(
        [row],
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )

    assert summary["outcome_distribution"] == {"revise": 1}
    assert summary["overlap_reaudit_required_count"] == 1
    assert summary["promotable_to_pass2_count"] == 0


def _clear_reaudit_report(row: dict) -> dict:
    return {
        "reaudit_version": "benchmark-revision-overlap-reaudit-v0.1",
        "candidate_id": row["candidate_id"],
        "original_question": row["question"],
        "revised_question": row["pass1_final_question"],
        "independence_unit_id": row["independence_unit_id"],
        "reaudit_decision": "clear",
        "reaudit_reasons": [],
        "parent_artifacts": {
            "queue_sha256": "a" * 64,
            "overlap_audit_sha256": "b" * 64,
        },
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
    }


def test_clear_reaudit_makes_revised_question_promotable_without_erasing_history(
    tmp_path,
):
    module = _load_module()
    row = _completed_pass1(
        module,
        pass1_outcome="revise",
        pass1_final_question="儿童肺炎治疗四十八小时后应如何复评？",
    )
    report = _clear_reaudit_report(row)
    report_path = tmp_path / "reaudit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    updated, summary = module.apply_pass1_reaudit(
        [row],
        report,
        report_path,
        module._compute_sha256(report_path),
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )

    assert updated[0]["pass1_outcome"] == "revise"
    assert updated[0]["pass1_overlap_reaudit_status"] == "clear"
    assert summary["overlap_reaudit_required_count"] == 0
    assert summary["promotable_to_pass2_count"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"candidate_id": "PMSQA-BV1C-other"}, "candidate_id"),
        ({"revised_question": "被篡改的问题"}, "修订问题"),
        ({"independence_unit_id": "IU-other"}, "独立性单元"),
        ({"reaudit_decision": "needs_review"}, "尚未通过"),
    ],
)
def test_pass1_reaudit_fails_closed_on_mismatch_or_unresolved(
    change,
    message,
    tmp_path,
):
    module = _load_module()
    row = _completed_pass1(
        module,
        pass1_outcome="revise",
        pass1_final_question="儿童肺炎治疗四十八小时后应如何复评？",
    )
    report = _clear_reaudit_report(row)
    report.update(change)
    report_path = tmp_path / "reaudit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.apply_pass1_reaudit(
            [row],
            report,
            report_path,
            module._compute_sha256(report_path),
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_reject_is_not_promotable_to_pass2():
    module = _load_module()
    row = _completed_pass1(module, pass1_outcome="reject")

    summary = module.validate_completed_pass1(
        [row],
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )

    assert summary["outcome_distribution"] == {"reject": 1}
    assert summary["promotable_to_pass2_count"] == 0


def test_real_assets_prepare_144_pending_rows_without_model_calls():
    module = _load_module()
    config = module.load_annotation_config(CONFIG_PATH)
    candidates = module.load_jsonl(CANDIDATE_PATH)
    anchors = module.load_jsonl(ANCHOR_PATH)
    manifest = module.load_json(MANIFEST_PATH)

    rows = module.build_pass1_queue(candidates, anchors, manifest, config)
    summary = module.summarize_pass1_queue(rows, config)

    assert len(rows) == 144
    assert len({row["candidate_id"] for row in rows}) == 144
    assert len({row["independence_unit_id"] for row in rows}) == 72
    assert summary["pending_count"] == 144
    assert summary["completed_count"] == 0
    assert summary["external_model_calls"] == 0
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0
    assert summary["estimated_cost"] == 0


def test_partial_pass1_progress_validates_completed_and_keeps_pending_rows():
    module = _load_module()
    second_candidate = _candidate(
        candidate_id="PMSQA-BV1C-test-002",
        question="儿童肺炎治疗后何时需要复评？",
        independence_unit_id="IU-test-002",
    )
    candidates = [_candidate(), second_candidate]
    config = _config(expected_candidate_count=2, expected_independence_unit_count=2)
    rows = module.build_pass1_queue(candidates, [_anchor()], _manifest(), config)
    rows[0].update(_pass1_decision(module))

    summary = module.validate_pass1_progress(
        rows,
        candidates,
        [_anchor()],
        _manifest(),
        config,
    )

    assert summary["completed_count"] == 1
    assert summary["pending_count"] == 1
    assert summary["outcome_distribution"] == {"accepted": 1}
    assert summary["promotable_to_pass2_count"] == 1


def test_partial_pass1_progress_rejects_partially_filled_pending_row():
    module = _load_module()
    rows = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )
    rows[0]["pass1_review_reason"] = "只填写了一个字段。"

    with pytest.raises(ValueError, match="部分填写"):
        module.validate_pass1_progress(
            rows,
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


@pytest.mark.parametrize(
    ("decisions", "message"),
    [
        (
            [
                {"candidate_id": "unknown"},
            ],
            "未知",
        ),
        (
            [
                {"candidate_id": "PMSQA-BV1C-test-001"},
                {"candidate_id": "PMSQA-BV1C-test-001"},
            ],
            "重复",
        ),
    ],
)
def test_apply_pass1_batch_rejects_unknown_or_duplicate_ids(decisions, message):
    module = _load_module()
    rows = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )

    with pytest.raises(ValueError, match=message):
        module.apply_pass1_batch(
            rows,
            decisions,
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_apply_pass1_batch_refuses_to_overwrite_reviewed_row():
    module = _load_module()
    row = _completed_pass1(module)

    with pytest.raises(ValueError, match="已完成"):
        module.apply_pass1_batch(
            [row],
            [_pass1_decision(module)],
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def test_apply_pass1_batch_only_changes_target_and_marks_revise_for_reaudit():
    module = _load_module()
    second_candidate = _candidate(
        candidate_id="PMSQA-BV1C-test-002",
        question="儿童肺炎治疗后何时需要复评？",
        independence_unit_id="IU-test-002",
    )
    candidates = [_candidate(), second_candidate]
    config = _config(expected_candidate_count=2, expected_independence_unit_count=2)
    rows = module.build_pass1_queue(candidates, [_anchor()], _manifest(), config)
    untouched = dict(rows[1])
    decision = _pass1_decision(
        module,
        pass1_outcome="revise",
        pass1_final_question="儿童肺炎治疗四十八小时后是否需要复评？",
    )

    updated, summary = module.apply_pass1_batch(
        rows,
        [decision],
        candidates,
        [_anchor()],
        _manifest(),
        config,
    )

    assert rows[0]["pass1_outcome"] == ""
    assert updated[1] == untouched
    assert summary["completed_count"] == 1
    assert summary["overlap_reaudit_required_count"] == 1
    assert summary["promotable_to_pass2_count"] == 0


def test_prepare_refuses_to_overwrite_reviewed_queue(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    candidate_path = tmp_path / "candidates.jsonl"
    anchor_path = tmp_path / "anchors.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "reviewed.csv"
    summary_path = tmp_path / "summary.json"
    config_path.write_text(
        json.dumps(_config(), ensure_ascii=False),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_candidate(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    anchor_path.write_text(
        json.dumps(_anchor(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(_manifest(), ensure_ascii=False),
        encoding="utf-8",
    )
    module.write_pass1_queue([_completed_pass1(module)], output_path)

    with pytest.raises(ValueError, match="不能覆盖"):
        module.run_prepare_pass1(
            config_path,
            candidate_path,
            anchor_path,
            manifest_path,
            output_path,
            summary_path,
        )


def test_run_apply_pass1_batch_writes_queue_and_progress_summary(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    candidate_path = tmp_path / "candidates.jsonl"
    anchor_path = tmp_path / "anchors.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "queue.csv"
    batch_path = tmp_path / "batch.json"
    progress_path = tmp_path / "progress.json"
    config_path.write_text(json.dumps(_config(), ensure_ascii=False), encoding="utf-8")
    candidate_path.write_text(
        json.dumps(_candidate(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    anchor_path.write_text(
        json.dumps(_anchor(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(_manifest(), ensure_ascii=False), encoding="utf-8")
    module.write_pass1_queue(
        module.build_pass1_queue(
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        ),
        output_path,
    )
    batch_path.write_text(
        json.dumps(
            {
                "batch_id": "pass1-batch-test",
                "annotation_version": "benchmark-annotation-v0.1",
                "dataset_version": "benchmark-v1.0-pass1-pending",
                "kb_version": "KB-medium-v1",
                "records": [_pass1_decision(module)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = module.run_apply_pass1_batch(
        config_path,
        candidate_path,
        anchor_path,
        manifest_path,
        output_path,
        batch_path,
        progress_path,
    )

    assert module.read_pass1_queue(output_path)[0]["pass1_outcome"] == "accepted"
    assert summary["completed_count"] == 1
    assert summary["pending_count"] == 0
    assert summary["applied_batches"] == ["pass1-batch-test"]
    assert json.loads(progress_path.read_text(encoding="utf-8"))["current_queue_sha256"]


def test_run_apply_pass1_reaudit_checks_parent_queue_and_writes_progress(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    candidate_path = tmp_path / "candidates.jsonl"
    anchor_path = tmp_path / "anchors.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "queue.csv"
    report_path = tmp_path / "reaudit.json"
    progress_path = tmp_path / "progress.json"
    config_path.write_text(json.dumps(_config(), ensure_ascii=False), encoding="utf-8")
    candidate_path.write_text(
        json.dumps(_candidate(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    anchor_path.write_text(
        json.dumps(_anchor(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(_manifest(), ensure_ascii=False), encoding="utf-8")
    row = _completed_pass1(
        module,
        pass1_outcome="revise",
        pass1_final_question="儿童肺炎治疗四十八小时后应如何复评？",
    )
    module.write_pass1_queue([row], output_path)
    report = _clear_reaudit_report(row)
    report["parent_artifacts"]["queue_sha256"] = module._compute_sha256(output_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = module.run_apply_pass1_reaudit(
        config_path,
        candidate_path,
        anchor_path,
        manifest_path,
        output_path,
        report_path,
        progress_path,
    )

    updated = module.read_pass1_queue(output_path)[0]
    assert updated["pass1_outcome"] == "revise"
    assert updated["pass1_overlap_reaudit_status"] == "clear"
    assert summary["overlap_reaudit_required_count"] == 0
    assert summary["promotable_to_pass2_count"] == 1
    assert summary["applied_reaudits"][0]["candidate_id"] == row["candidate_id"]


def test_pass1_review_metadata_rejects_garbled_text():
    module = _load_module()
    row = _completed_pass1(
        module,
        pass1_required_claims=["????????"],
    )

    with pytest.raises(ValueError, match="乱码或编码损坏"):
        module.validate_pass1_progress(
            [row],
            [_candidate()],
            [_anchor()],
            _manifest(),
            _config(),
        )


def _pass1_correction(module, row, **overrides) -> dict:
    correction = {
        "correction_id": "pass1-correction-test-001",
        "annotation_version": "benchmark-annotation-v0.1",
        "dataset_version": "benchmark-v1.0-pass1-pending",
        "kb_version": "KB-medium-v1",
        "parent_queue_sha256": "",
        "candidate_id": row["candidate_id"],
        "field": "pass1_review_reason",
        "expected_old_value": row["pass1_review_reason"],
        "corrected_value": "已核对正确的资料标题、页码与证据边界。",
        "reason": "修正人工核验理由中的资料标题笔误，不改变证据或决策。",
    }
    correction.update(overrides)
    return correction


def test_apply_pass1_correction_updates_only_whitelisted_field():
    module = _load_module()
    row = _completed_pass1(module)
    before = dict(row)

    updated = module.apply_pass1_correction(
        [row],
        _pass1_correction(module, row),
        _config(),
    )

    changed = {
        key
        for key in before
        if before[key] != updated[0][key]
    }
    assert changed == {"pass1_review_reason"}
    assert updated[0]["pass1_review_reason"] == "已核对正确的资料标题、页码与证据边界。"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"field": "pass1_expected_decision"}, "不允许纠正字段"),
        ({"expected_old_value": "并不存在的旧值"}, "当前值不一致"),
    ],
)
def test_apply_pass1_correction_rejects_unsafe_change(overrides, error):
    module = _load_module()
    row = _completed_pass1(module)

    with pytest.raises(ValueError, match=error):
        module.apply_pass1_correction(
            [row],
            _pass1_correction(module, row, **overrides),
            _config(),
        )


def test_apply_pass1_correction_rejects_pending_candidate():
    module = _load_module()
    row = module.build_pass1_queue(
        [_candidate()],
        [_anchor()],
        _manifest(),
        _config(),
    )[0]

    with pytest.raises(ValueError, match="尚未完成首轮核验"):
        module.apply_pass1_correction(
            [row],
            _pass1_correction(
                module,
                row,
                expected_old_value="",
            ),
            _config(),
        )


def test_run_apply_pass1_correction_checks_hash_and_records_audit(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "queue.csv"
    correction_path = tmp_path / "correction.json"
    progress_path = tmp_path / "progress.json"
    config_path.write_text(json.dumps(_config(), ensure_ascii=False), encoding="utf-8")
    row = _completed_pass1(module)
    module.write_pass1_queue([row], output_path)
    correction = _pass1_correction(
        module,
        row,
        parent_queue_sha256=module._compute_sha256(output_path),
    )
    correction_path.write_text(
        json.dumps(correction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = module.run_apply_pass1_correction(
        config_path,
        output_path,
        correction_path,
        progress_path,
    )

    updated = module.read_pass1_queue(output_path)[0]
    assert updated["pass1_review_reason"] == correction["corrected_value"]
    assert summary["applied_corrections"][0]["correction_id"] == correction["correction_id"]
    assert summary["applied_corrections"][0]["candidate_id"] == row["candidate_id"]
    assert summary["previous_queue_sha256"] == correction["parent_queue_sha256"]
    assert summary["current_queue_sha256"] != correction["parent_queue_sha256"]

    with pytest.raises(ValueError, match="已应用"):
        module.run_apply_pass1_correction(
            config_path,
            output_path,
            correction_path,
            progress_path,
        )


def test_run_apply_pass1_correction_rejects_parent_hash_mismatch(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "queue.csv"
    correction_path = tmp_path / "correction.json"
    progress_path = tmp_path / "progress.json"
    config_path.write_text(json.dumps(_config(), ensure_ascii=False), encoding="utf-8")
    row = _completed_pass1(module)
    module.write_pass1_queue([row], output_path)
    correction_path.write_text(
        json.dumps(
            _pass1_correction(module, row, parent_queue_sha256="0" * 64),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="父队列 SHA-256"):
        module.run_apply_pass1_correction(
            config_path,
            output_path,
            correction_path,
            progress_path,
        )
