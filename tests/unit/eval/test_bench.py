"""Bench helpers: percentiles and cache-disabling config overrides."""

import pytest

from docseek.eval.bench import cold, override, percentiles, uncached_queries


def test_percentiles_hand_computed() -> None:
    p = percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert p["p50"] == 30.0 and p["n"] == 5
    assert p["p95"] == pytest.approx(48.0)  # linear interpolation: 40 + 0.8 * 10
    assert p["p99"] == pytest.approx(49.6)


def test_override_reaches_nested_blocks() -> None:
    raw = {
        "type": "rerank",
        "top_n": 20,
        "first_stage": {
            "type": "hybrid",
            "children": {
                "bm25": {"type": "bm25", "k1": 1.5},
                "dense": {"type": "dense", "model": "m"},
            },
        },
    }
    c = override(raw, cold)
    assert c["cache_path"] is None
    assert c["first_stage"]["children"]["bm25"]["index_dir"] is None
    assert c["first_stage"]["children"]["dense"] == {
        "type": "dense",
        "model": "m",
        "cache_dir": None,
        "location": ":memory:",
    }
    live = override(raw, uncached_queries)["first_stage"]["children"]["dense"]
    assert live["cache_dir"] is None and "location" not in live  # production Qdrant
    assert raw["first_stage"]["children"]["dense"] == {"type": "dense", "model": "m"}  # pure
