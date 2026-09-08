import json
from pathlib import Path

import pytest

from commercereclab.evaluation.final import build_final_selection, write_report


def _dump(root: Path, directory: str, filename: str, payload: dict):
    d = root / directory
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps(payload))


def _artifacts(root: Path):
    metrics = {
        "test": {
            "category_conditioned_popularity": {
                "view": {"ndcg_at_k": 0.12},
                "addtocart": {"ndcg_at_k": 0.15},
                "transaction": {"ndcg_at_k": 0.16},
            }
        }
    }
    _dump(root, "v0_3_behavioral_baselines", "behavioral_baselines.json", {"metrics": metrics})
    _dump(root, "v0_4_hybrid_ablation", "hybrid_ablation.json", {
        "validation_retain_full_hybrid": False,
        "validation_mean_ndcg": {"hybrid_full": 0.10, "category_only": 0.17},
    })
    _dump(root, "v0_5_learned_ranker", "learned_ranker.json", {
        "validation_retain_learned_ranker": False,
        "validation_mean_ndcg": {"learned_task_specific": 0.14, "category_only": 0.17},
    })
    _dump(root, "v0_6_candidate_retrieval", "candidate_retrieval.json", {
        "validation_retain_expanded_union": True,
        "validation_mean_candidate_recall": {"base_union": 0.61, "expanded_union": 0.72},
    })
    candidate_recall = {
        "validation": {"c300_b100_p0": {"view": .67, "addtocart": .73, "transaction": .71}},
        "test": {"c300_b100_p0": {"view": .62, "addtocart": .64, "transaction": .64}},
    }
    candidate_size = {
        "test": {
            "c300_b100_p0": {"mean": 319.0},
            "c300_b300_p300": {"mean": 690.0},
        }
    }
    _dump(root, "v0_7_retrieval_efficiency", "retrieval_efficiency.json", {
        "selected_efficient_configuration": "c300_b100_p0",
        "selected_test_retention": {"view": .966, "addtocart": .970, "transaction": .977},
        "candidate_recall": candidate_recall,
        "candidate_size": candidate_size,
    })
    _dump(root, "v0_8_efficient_reranker", "efficient_reranker.json", {
        "validation_retain_learned_ranker": False,
        "validation_mean_ndcg": {"learned_task_specific": .148, "category_only": .169},
        "metrics": {"test": {"learned_task_specific": {"view": {"ndcg_at_k": .135}}}},
    })
    _dump(root, "v0_9_stage_aware_ranking", "stage_aware_ranking.json", {
        "validation_retain_stage_conditioning": False,
        "validation_mean_ndcg": {"stage_conditioned": .150, "category_only": .178},
    })


def test_final_selection_builds_and_writes(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _artifacts(artifacts)
    result = build_final_selection(artifacts)
    assert result.retrieval_configuration.startswith("c300_b100_p0")
    assert [row.decision for row in result.benchmark].count("reject") == 4
    json_path, md_path = write_report(result, tmp_path / "out")
    data = json.loads(json_path.read_text())
    assert data["evidence_summary"]["selected_retriever_test_mean_candidates"] == 319.0
    text = md_path.read_text()
    assert "retrieval and category context earned their complexity" in text
    assert "c300_b100_p0" in text


def test_final_selection_fails_on_decision_drift(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _artifacts(artifacts)
    p = artifacts / "v0_8_efficient_reranker" / "efficient_reranker.json"
    data = json.loads(p.read_text())
    data["validation_retain_learned_ranker"] = True
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="v0.8 frozen decision"):
        build_final_selection(artifacts)
