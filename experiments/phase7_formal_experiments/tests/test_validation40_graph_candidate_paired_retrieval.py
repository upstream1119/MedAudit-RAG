from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_graph_candidate_paired_retrieval.py"
)
GRAPH_MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "graph_candidate_expansion.py"
)
V03_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "validation40_graph_candidate_paired_retrieval_v0_3.json"
)
V04_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "validation40_graph_candidate_paired_retrieval_v0_4.json"
)
V05_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "validation40_graph_candidate_paired_retrieval_v0_5.json"
)


def _load_module(path: Path, name: str):
    assert path.exists(), f"Missing module: {path.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    candidate_key: str,
    content: str,
    *,
    source_file: str = "A.pdf",
    page_number: int = 1,
    rrf_rank: int = 1,
) -> dict:
    collection, document_id = candidate_key.split("::", maxsplit=1)
    return {
        "candidate_key": candidate_key,
        "collection": collection,
        "document_id": document_id,
        "content": content,
        "source_file": source_file,
        "page_number": page_number,
        "granularity": 128,
        "rrf_rank": rrf_rank,
        "rrf_score": 1.0 / (60 + rrf_rank),
        "route_traces": [],
    }


class _FakeScorer:
    def __init__(self):
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return np.asarray(
            [10.0 if "图证据" in pair[1] else float(len(pair[1])) for pair in pairs],
            dtype=np.float32,
        )


def _graph_index():
    graph_module = _load_module(GRAPH_MODULE_PATH, "graph_candidate_expansion_test")
    return graph_module.build_candidate_graph_index(
        [
            _candidate(
                "detail_128::graph-1",
                "图证据：治疗48-72小时症状无改善时，应再次进行临床或实验室评估。",
                source_file="指南.pdf",
                page_number=26,
            )
        ]
    )


def _source_routing_inputs(graph_index: dict) -> tuple[dict, dict, dict]:
    router = __import__(
        "experiments.phase7_formal_experiments.runtime_graph_path_router",
        fromlist=["build_runtime_path_catalog"],
    )
    lexicon = {
        "lexicon_version": "fixture-v0.2",
        "entries": [
            {
                "constraint_type": "clinical_condition",
                "normalized_value": "mycoplasma_pneumoniae_pneumonia",
                "aliases": ["MPP"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication_class",
                "normalized_value": "corticosteroid",
                "aliases": ["糖皮质激素"],
                "strong_anchor": True,
            },
        ],
    }
    policy = {
        "allow_specific_condition_class_path": True,
        "max_total_paths": 20,
        "max_paths_per_source": 2,
        "max_paths_per_source_page": 1,
    }
    return router.build_runtime_path_catalog(graph_index, lexicon), lexicon, policy


def test_v03_config_freezes_matched_compute_top24_from_top40_source():
    config = json.loads(V03_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["phase"] == "Phase 7-C1c-4e-3a"
    assert config["source_method"] == "dense_exact_sparse_rrf"
    assert config["source_budget"] == 40
    assert config["candidate_budget"] == 24
    assert config["graph_quota"] == 4
    assert config["final_evidence_k"] == 4
    assert config["f_method"] == "f24_exact_hybrid_reranker_dedup"
    assert config["g1_method"] == "g1_v0_3_source_routed_union24_reranker_dedup"
    assert config["execution_guards"]["gold_access"] is False
    assert config["execution_guards"]["pilot_test_content_access"] is False
    assert config["execution_guards"]["external_model_calls"] is False


def test_v04_config_preserves_top20_and_limits_graph_to_four_reserve_slots():
    config = json.loads(V04_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["phase"] == "Phase 7-C1c-4e-3a"
    assert config["source_budget"] == 40
    assert config["baseline_prefix_budget"] == 20
    assert config["candidate_budget"] == 24
    assert config["graph_quota"] == 4
    assert config["final_evidence_k"] == 4
    assert config["f_method"] == "f24_prefix_stable_hybrid_reranker_dedup"
    assert (
        config["g1_method"]
        == "g1_v0_4_prefix_stable_graph_expand_reranker_dedup"
    )
    assert config["execution_guards"]["gold_access"] is False
    assert config["execution_guards"]["pilot_test_content_access"] is False
    assert config["execution_guards"]["external_model_calls"] is False


def test_v05_config_adds_provenance_without_changing_frozen_methods():
    config = json.loads(V05_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["phase"] == "Phase 7-C1c-4e-3a-provenance"
    assert config["source_budget"] == 40
    assert config["baseline_prefix_budget"] == 20
    assert config["candidate_budget"] == 24
    assert config["graph_quota"] == 4
    assert config["f_method"] == "f24_prefix_stable_hybrid_reranker_dedup"
    assert (
        config["g1_method"]
        == "g1_v0_4_prefix_stable_graph_expand_reranker_dedup"
    )
    assert (
        config["graph_path_trace_version"]
        == "phase7-runtime-graph-path-trace-v0.1"
    )
    assert config["execution_guards"]["gold_access"] is False
    assert config["execution_guards"]["pilot_test_content_access"] is False
    assert config["execution_guards"]["external_model_calls"] is False


def test_paired_sample_routes_graph_candidates_without_changing_f_pool():
    module = _load_module(MODULE_PATH, "paired_retrieval_source_route")
    graph_module = _load_module(GRAPH_MODULE_PATH, "graph_candidate_source_route")
    baseline = [
        _candidate(
            f"detail_128::base-{index}",
            f"基线证据{index}",
            page_number=index,
            rrf_rank=index,
        )
        for index in range(1, 5)
    ]
    graph_index = graph_module.build_candidate_graph_index(
        [
            _candidate(
                "detail_128::graph-specific",
                "图证据：MPP在限定情况下可考虑糖皮质激素。",
                source_file="MPP诊疗指南.pdf",
                page_number=10,
            )
        ]
    )
    catalog, lexicon, policy = _source_routing_inputs(graph_index)

    row = module.build_paired_sample(
        sample_id="S1",
        question="MPP糖皮质激素治疗是否有依据？",
        baseline_candidates=baseline,
        graph_index=graph_index,
        scorer=_FakeScorer(),
        total_budget=4,
        graph_quota=1,
        final_evidence_k=2,
        reranker_batch_size=2,
        dedup_ngram_size=3,
        dedup_overlap_threshold=0.8,
        runtime_path_catalog=catalog,
        runtime_lexicon=lexicon,
        routing_policy=policy,
    )

    f_candidates = row["methods"][module.F_METHOD]["candidates_top20"]
    g1_candidates = row["methods"][module.G1_METHOD]["candidates_top20"]
    assert {item["candidate_key"] for item in f_candidates} == {
        f"detail_128::base-{index}" for index in range(1, 5)
    }
    assert len(g1_candidates) == 4
    assert row["graph_expansion_audit"]["added_candidate_keys"] == [
        "detail_128::graph-specific"
    ]
    assert row["graph_expansion_audit"]["graph_candidate_count"] == 1
    graph_candidate = next(
        item
        for item in g1_candidates
        if item["candidate_origin"] == "graph_expansion"
    )
    assert graph_candidate["graph_path_trace"]["candidate"]["candidate_key"] == (
        graph_candidate["candidate_key"]
    )
    assert row["graph_expansion_audit"]["graph_trace_required"] is True
    assert row["graph_expansion_audit"]["graph_trace_count"] == 1
    assert row["graph_expansion_audit"]["graph_trace_complete"] is True
    assert len(row["graph_expansion_audit"]["graph_trace_sha256"]) == 64


def test_graph_path_trace_validation_fails_closed_on_candidate_mismatch():
    module = _load_module(MODULE_PATH, "paired_retrieval_trace_guard")
    candidate = _candidate(
        "detail_128::graph-specific",
        "图证据：MPP在限定情况下可考虑糖皮质激素。",
        source_file="MPP诊疗指南.pdf",
        page_number=10,
    )
    candidate["candidate_origin"] = "graph_expansion"
    candidate["graph_path_trace"] = {
        "trace_version": "phase7-runtime-graph-path-trace-v0.1",
        "router_version": "phase7-runtime-graph-path-router-v0.1",
        "route_decision": "selected",
        "raw_rank": 1,
        "route_rank": 1,
        "source_condition_tier": 0,
        "source_condition_tier_label": "source_condition_exact",
        "query_constraints": [],
        "matched_constraints": [],
        "content_matched_constraints": [],
        "source_matched_constraints": [],
        "candidate": {
            "candidate_key": "detail_128::different",
            "source_file": "MPP诊疗指南.pdf",
            "page_number": 10,
        },
    }

    with pytest.raises(ValueError, match="candidate identity mismatch"):
        module._validate_graph_path_trace(candidate)


def test_same_budget_graph_quota_replaces_instead_of_expanding():
    module = _load_module(MODULE_PATH, "paired_retrieval_budget")
    baseline = [
        _candidate(
            f"detail_128::base-{index}",
            f"基线证据{index}",
            page_number=index,
            rrf_rank=index,
        )
        for index in range(1, 5)
    ]
    scorer = _FakeScorer()

    row = module.build_paired_sample(
        sample_id="S1",
        question="治疗48-72小时症状无改善时，是否需要再次评估？",
        baseline_candidates=baseline,
        graph_index=_graph_index(),
        scorer=scorer,
        total_budget=4,
        graph_quota=1,
        final_evidence_k=2,
        reranker_batch_size=2,
        dedup_ngram_size=3,
        dedup_overlap_threshold=0.8,
    )

    f_candidates = row["methods"]["f_exact_hybrid_reranker_dedup"]["candidates_top20"]
    g_candidates = row["methods"]["g1_exact_graph_expand_reranker_dedup"]["candidates_top20"]
    assert len(f_candidates) == len(g_candidates) == 4
    assert sum(item["candidate_origin"] == "graph_expansion" for item in g_candidates) == 1
    g_candidate_keys = {item["candidate_key"] for item in g_candidates}
    assert {
        "detail_128::base-1",
        "detail_128::base-2",
        "detail_128::base-3",
    }.issubset(g_candidate_keys)
    assert "detail_128::base-4" not in g_candidate_keys
    assert len(scorer.calls) == 2


def test_no_graph_match_keeps_same_candidate_keys_for_both_methods():
    module = _load_module(MODULE_PATH, "paired_retrieval_no_match")
    baseline = [
        _candidate("detail_128::base-1", "儿童肺炎定义。"),
        _candidate("detail_128::base-2", "一般评估。", rrf_rank=2),
    ]

    row = module.build_paired_sample(
        sample_id="S1",
        question="儿童肺炎的一般原则有哪些？",
        baseline_candidates=baseline,
        graph_index=_graph_index(),
        scorer=_FakeScorer(),
        total_budget=2,
        graph_quota=1,
        final_evidence_k=2,
        reranker_batch_size=2,
        dedup_ngram_size=3,
        dedup_overlap_threshold=0.8,
    )

    methods = row["methods"]
    f_keys = [
        item["candidate_key"]
        for item in methods["f_exact_hybrid_reranker_dedup"]["candidates_top20"]
    ]
    g_keys = [
        item["candidate_key"]
        for item in methods["g1_exact_graph_expand_reranker_dedup"]["candidates_top20"]
    ]
    assert f_keys == g_keys
    assert row["graph_expansion_audit"]["added_candidate_keys"] == []


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner_fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "runtime.jsonl"
    exact = tmp_path / "exact.jsonl"
    exact_audit = tmp_path / "exact_audit.json"
    graph_index_path = tmp_path / "graph_index.json"
    graph_manifest = tmp_path / "graph_manifest.json"
    pilot = tmp_path / "pilot.jsonl"
    question = "治疗48-72小时症状无改善时，是否需要再次评估？"
    runtime_rows = [
        {
            "sample_id": "S1",
            "question": question,
            "dataset_version": "dataset-v1",
            "kb_version": "kb-v1",
        }
    ]
    baseline = [
        _candidate(
            f"detail_128::base-{index}",
            f"基线证据{index}",
            page_number=index,
            rrf_rank=index,
        )
        for index in range(1, 5)
    ]
    exact_rows = [
        {
            "sample_id": "S1",
            "question": question,
            "dataset_version": "dataset-v1",
            "kb_version": "kb-v1",
            "methods": {"dense_exact_sparse_rrf": {"4": baseline}},
        }
    ]
    _write_jsonl(runtime, runtime_rows)
    _write_jsonl(exact, exact_rows)
    _write_json(exact_audit, {"ready": True})
    _write_json(graph_index_path, _graph_index())
    _write_json(
        graph_manifest,
        {
            "ready": True,
            "files": {
                "graph_index": {
                    "path": graph_index_path.name,
                    "sha256": _sha(graph_index_path),
                }
            },
        },
    )
    pilot.write_bytes(b"must-not-be-parsed")
    paths = {
        "runtime": runtime,
        "exact": exact,
        "exact_audit": exact_audit,
        "graph_index": graph_index_path,
        "graph_manifest": graph_manifest,
        "pilot": pilot,
    }
    config = {
        "config_version": "test-v1",
        "dataset_version": "dataset-v1",
        "kb_version": "kb-v1",
        "expected_count": 1,
        "expected_runtime_projection_sha256": _sha(runtime),
        "expected_exact_results_sha256": _sha(exact),
        "expected_exact_audit_sha256": _sha(exact_audit),
        "expected_graph_manifest_sha256": _sha(graph_manifest),
        "expected_pilot_test_sha256": _sha(pilot),
        "source_method": "dense_exact_sparse_rrf",
        "source_budget": 4,
        "candidate_budget": 4,
        "graph_quota": 1,
        "final_evidence_k": 2,
        "reranker_batch_size": 2,
        "dedup_ngram_size": 3,
        "dedup_overlap_threshold": 0.8,
        "results_filename": "results.jsonl",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
    }
    return config, paths


def test_runner_fails_on_hash_drift_before_loading_scorer_or_writing(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_hash")
    config, paths = _runner_fixture(tmp_path)
    config["expected_exact_results_sha256"] = "0" * 64
    scorer_loaded = False

    def scorer_factory():
        nonlocal scorer_loaded
        scorer_loaded = True
        return _FakeScorer()

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="exact results SHA-256 mismatch"):
        module.run_paired_retrieval(
            runtime_projection_path=paths["runtime"],
            exact_results_path=paths["exact"],
            exact_audit_path=paths["exact_audit"],
            graph_manifest_path=paths["graph_manifest"],
            pilot_test_path=paths["pilot"],
            output_dir=output,
            config=config,
            scorer_factory=scorer_factory,
        )
    assert scorer_loaded is False
    assert not output.exists()


def test_runner_rejects_gold_only_fields_and_sample_mismatch(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_guards")
    config, paths = _runner_fixture(tmp_path)
    rows = [json.loads(line) for line in paths["exact"].read_text(encoding="utf-8").splitlines()]
    rows[0]["methods"]["dense_exact_sparse_rrf"]["4"][0]["expected_decision"] = "answer"
    _write_jsonl(paths["exact"], rows)
    config["expected_exact_results_sha256"] = _sha(paths["exact"])

    with pytest.raises(ValueError, match="gold-only"):
        module.run_paired_retrieval(
            runtime_projection_path=paths["runtime"],
            exact_results_path=paths["exact"],
            exact_audit_path=paths["exact_audit"],
            graph_manifest_path=paths["graph_manifest"],
            pilot_test_path=paths["pilot"],
            output_dir=tmp_path / "gold-output",
            config=config,
            scorer_factory=_FakeScorer,
        )

    config, paths = _runner_fixture(tmp_path / "second")
    paths["runtime"].parent.mkdir(parents=True, exist_ok=True)
    runtime_rows = [json.loads(line) for line in paths["runtime"].read_text(encoding="utf-8").splitlines()]
    runtime_rows[0]["sample_id"] = "DIFFERENT"
    _write_jsonl(paths["runtime"], runtime_rows)
    config["expected_runtime_projection_sha256"] = _sha(paths["runtime"])
    with pytest.raises(ValueError, match="sample_id/question mismatch"):
        module.run_paired_retrieval(
            runtime_projection_path=paths["runtime"],
            exact_results_path=paths["exact"],
            exact_audit_path=paths["exact_audit"],
            graph_manifest_path=paths["graph_manifest"],
            pilot_test_path=paths["pilot"],
            output_dir=tmp_path / "mismatch-output",
            config=config,
            scorer_factory=_FakeScorer,
        )


def test_two_runs_have_byte_identical_core_results_and_never_parse_pilot(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_determinism")
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    config, paths = _runner_fixture(fixture_root)
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    first = module.run_paired_retrieval(
        runtime_projection_path=paths["runtime"],
        exact_results_path=paths["exact"],
        exact_audit_path=paths["exact_audit"],
        graph_manifest_path=paths["graph_manifest"],
        pilot_test_path=paths["pilot"],
        output_dir=run_a,
        config=config,
        scorer_factory=_FakeScorer,
    )
    second = module.run_paired_retrieval(
        runtime_projection_path=paths["runtime"],
        exact_results_path=paths["exact"],
        exact_audit_path=paths["exact_audit"],
        graph_manifest_path=paths["graph_manifest"],
        pilot_test_path=paths["pilot"],
        output_dir=run_b,
        config=config,
        scorer_factory=_FakeScorer,
    )

    assert (run_a / "results.jsonl").read_bytes() == (run_b / "results.jsonl").read_bytes()
    assert first["audit"]["pilot_test_accessed"] is False
    assert second["audit"]["gold_accessed"] is False
    assert first["audit"]["external_model_calls"] == 0
    assert first["audit"]["input_tokens"] == first["audit"]["output_tokens"] == 0


def test_runner_stably_truncates_deeper_source_pool_to_candidate_budget(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_deeper_source")
    config, paths = _runner_fixture(tmp_path)
    exact_rows = [
        json.loads(line)
        for line in paths["exact"].read_text(encoding="utf-8").splitlines()
    ]
    deeper_pool = [
        _candidate(
            f"detail_128::deep-{index}",
            f"深层候选{index}",
            page_number=index,
            rrf_rank=index,
        )
        for index in range(1, 7)
    ]
    exact_rows[0]["methods"]["dense_exact_sparse_rrf"]["6"] = deeper_pool
    _write_jsonl(paths["exact"], exact_rows)
    config["source_budget"] = 6
    config["candidate_budget"] = 4
    config["graph_quota"] = 0
    config["expected_exact_results_sha256"] = _sha(paths["exact"])

    output = module.run_paired_retrieval(
        runtime_projection_path=paths["runtime"],
        exact_results_path=paths["exact"],
        exact_audit_path=paths["exact_audit"],
        graph_manifest_path=paths["graph_manifest"],
        pilot_test_path=paths["pilot"],
        output_dir=tmp_path / "deeper-source-output",
        config=config,
        scorer_factory=_FakeScorer,
    )

    method = output["results"][0]["methods"]["f_exact_hybrid_reranker_dedup"]
    assert "candidates_top20" not in method
    assert [row["candidate_key"] for row in method["candidates_top4"]] == [
        f"detail_128::deep-{index}" for index in range(1, 5)
    ]


def test_runner_builds_prefix_stable_baseline_before_graph_replacement(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_prefix_stable")
    config, paths = _runner_fixture(tmp_path)
    exact_rows = [
        json.loads(line)
        for line in paths["exact"].read_text(encoding="utf-8").splitlines()
    ]
    frozen_prefix = exact_rows[0]["methods"]["dense_exact_sparse_rrf"]["4"]
    deeper_pool = [
        _candidate(
            f"detail_128::deep-{index}",
            f"深层候选{index}",
            page_number=index + 10,
            rrf_rank=index,
        )
        for index in range(1, 7)
    ]
    exact_rows[0]["methods"]["dense_exact_sparse_rrf"]["6"] = deeper_pool
    _write_jsonl(paths["exact"], exact_rows)
    config["source_budget"] = 6
    config["baseline_prefix_budget"] = 4
    config["candidate_budget"] = 6
    config["graph_quota"] = 1
    config["expected_exact_results_sha256"] = _sha(paths["exact"])

    output = module.run_paired_retrieval(
        runtime_projection_path=paths["runtime"],
        exact_results_path=paths["exact"],
        exact_audit_path=paths["exact_audit"],
        graph_manifest_path=paths["graph_manifest"],
        pilot_test_path=paths["pilot"],
        output_dir=tmp_path / "prefix-stable-output",
        config=config,
        scorer_factory=_FakeScorer,
    )

    row = output["results"][0]
    f_candidates = row["methods"]["f_exact_hybrid_reranker_dedup"][
        "candidates_top6"
    ]
    candidate_keys = {item["candidate_key"] for item in f_candidates}
    assert {item["candidate_key"] for item in frozen_prefix}.issubset(candidate_keys)
    assert {"detail_128::deep-1", "detail_128::deep-2"}.issubset(candidate_keys)
    g1_candidates = row["methods"]["g1_exact_graph_expand_reranker_dedup"][
        "candidates_top6"
    ]
    g1_keys = {item["candidate_key"] for item in g1_candidates}
    assert {item["candidate_key"] for item in frozen_prefix}.issubset(g1_keys)
    assert "detail_128::graph-1" in g1_keys
    assert row["graph_expansion_audit"]["baseline_prefix_preserved"] is True
    assert row["graph_expansion_audit"]["baseline_reserve_count"] == 2
    assert row["graph_expansion_audit"]["graph_candidate_count"] == 1


def test_runner_rejects_source_budget_smaller_than_candidate_budget(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_invalid_source_budget")
    config, paths = _runner_fixture(tmp_path)
    config["source_budget"] = 3
    config["candidate_budget"] = 4
    scorer_loaded = False

    def scorer_factory():
        nonlocal scorer_loaded
        scorer_loaded = True
        return _FakeScorer()

    with pytest.raises(ValueError, match="Source budget cannot be smaller"):
        module.run_paired_retrieval(
            runtime_projection_path=paths["runtime"],
            exact_results_path=paths["exact"],
            exact_audit_path=paths["exact_audit"],
            graph_manifest_path=paths["graph_manifest"],
            pilot_test_path=paths["pilot"],
            output_dir=tmp_path / "invalid-source-budget-output",
            config=config,
            scorer_factory=scorer_factory,
        )

    assert scorer_loaded is False


def test_runner_rejects_prefix_budget_larger_than_candidate_budget(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_invalid_prefix_budget")
    config, paths = _runner_fixture(tmp_path)
    config["baseline_prefix_budget"] = 5
    scorer_loaded = False

    def scorer_factory():
        nonlocal scorer_loaded
        scorer_loaded = True
        return _FakeScorer()

    with pytest.raises(ValueError, match="baseline_prefix_budget is invalid"):
        module.run_paired_retrieval(
            runtime_projection_path=paths["runtime"],
            exact_results_path=paths["exact"],
            exact_audit_path=paths["exact_audit"],
            graph_manifest_path=paths["graph_manifest"],
            pilot_test_path=paths["pilot"],
            output_dir=tmp_path / "invalid-prefix-budget-output",
            config=config,
            scorer_factory=scorer_factory,
        )

    assert scorer_loaded is False


def test_runner_registers_source_routed_v02_and_hashes_routing_inputs(tmp_path):
    module = _load_module(MODULE_PATH, "paired_retrieval_v02_runner")
    config, paths = _runner_fixture(tmp_path / "fixture-v02")
    question = "MPP糖皮质激素治疗是否有依据？"
    runtime_rows = [
        json.loads(line)
        for line in paths["runtime"].read_text(encoding="utf-8").splitlines()
    ]
    exact_rows = [
        json.loads(line)
        for line in paths["exact"].read_text(encoding="utf-8").splitlines()
    ]
    runtime_rows[0]["question"] = question
    exact_rows[0]["question"] = question
    _write_jsonl(paths["runtime"], runtime_rows)
    _write_jsonl(paths["exact"], exact_rows)
    graph_module = _load_module(GRAPH_MODULE_PATH, "graph_candidate_v02_runner")
    _write_json(
        paths["graph_index"],
        graph_module.build_candidate_graph_index(
            [
                _candidate(
                    "detail_128::graph-specific",
                    "图证据：MPP在限定情况下可考虑糖皮质激素。",
                    source_file="MPP诊疗指南.pdf",
                    page_number=10,
                )
            ]
        ),
    )
    _write_json(
        paths["graph_manifest"],
        {
            "ready": True,
            "files": {
                "graph_index": {
                    "path": paths["graph_index"].name,
                    "sha256": _sha(paths["graph_index"]),
                }
            },
        },
    )
    config["expected_runtime_projection_sha256"] = _sha(paths["runtime"])
    config["expected_exact_results_sha256"] = _sha(paths["exact"])
    config["expected_graph_manifest_sha256"] = _sha(paths["graph_manifest"])
    lexicon_path = tmp_path / "runtime_lexicon.json"
    routing_manifest_path = tmp_path / "routing_manifest.json"
    _write_json(
        lexicon_path,
        {
            "lexicon_version": "fixture-v0.2",
            "entries": [
                {
                    "constraint_type": "clinical_condition",
                    "normalized_value": "mycoplasma_pneumoniae_pneumonia",
                    "aliases": ["MPP"],
                    "strong_anchor": True,
                },
                {
                    "constraint_type": "medication_class",
                    "normalized_value": "corticosteroid",
                    "aliases": ["糖皮质激素"],
                    "strong_anchor": True,
                },
            ],
        },
    )
    _write_json(
        routing_manifest_path,
        {
            "router_version": "phase7-runtime-graph-path-router-v0.1",
            "ready": True,
        },
    )
    config.update(
        {
            "f_method": "f_exact_hybrid_reranker_dedup",
            "g1_method": "g1_v0_2_source_routed_graph_expand_reranker_dedup",
            "audit_version": "phase7-c1c4e2e2-paired-retrieval-audit-v0.2",
            "manifest_version": "phase7-c1c4e2e2-paired-retrieval-manifest-v0.2",
            "expected_runtime_lexicon_sha256": _sha(lexicon_path),
            "expected_routing_manifest_sha256": _sha(routing_manifest_path),
            "routing_policy": {
                "allow_specific_condition_class_path": True,
                "max_total_paths": 20,
                "max_paths_per_source": 2,
                "max_paths_per_source_page": 1,
            },
        }
    )

    result = module.run_paired_retrieval(
        runtime_projection_path=paths["runtime"],
        exact_results_path=paths["exact"],
        exact_audit_path=paths["exact_audit"],
        graph_manifest_path=paths["graph_manifest"],
        pilot_test_path=paths["pilot"],
        runtime_lexicon_path=lexicon_path,
        routing_manifest_path=routing_manifest_path,
        output_dir=tmp_path / "output-v02",
        config=config,
        scorer_factory=_FakeScorer,
    )

    methods = result["results"][0]["methods"]
    assert set(methods) == {config["f_method"], config["g1_method"]}
    assert result["results"][0]["graph_expansion_audit"]["graph_candidate_count"] == 1
    assert result["audit"]["runtime_lexicon_sha256"] == _sha(lexicon_path)
    assert result["audit"]["routing_manifest_sha256"] == _sha(
        routing_manifest_path
    )
    assert result["audit"]["audit_version"] == config["audit_version"]
    assert result["manifest"]["manifest_version"] == config["manifest_version"]
    assert result["audit"]["pilot_test_accessed"] is False
    assert result["audit"]["gold_accessed"] is False
