import json
from pathlib import Path

from docseek.eval.golden import GoldenQuery
from docseek.eval.pooling import (
    build_v2,
    judge_pairs,
    pool_sections,
    read_judgments,
    unlabelled_pairs,
    write_judgments,
)


def _run(tmp: Path, name: str, rows: dict[str, list[str]]) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    with (d / "per_query.jsonl").open("w", encoding="utf-8") as fh:
        for qid, ranked in rows.items():
            fh.write(json.dumps({"qid": qid, "ranked": ranked}) + "\n")
    return d


def test_pool_judge_and_merge(tmp_path: Path) -> None:
    q1 = GoldenQuery(
        qid="q0001",
        text="t",
        qtype="howto",
        relevant={"s1": 2},
        origin="synthetic",
        hard_negatives=["s9"],
    )
    q2 = GoldenQuery(qid="q0002", text="u", qtype="unanswerable", origin="synthetic")
    a = _run(tmp_path, "a", {"q0001": ["s1", "s2", "s9", "s4"], "q0002": ["s5"]})
    b = _run(tmp_path, "b", {"q0001": ["s3", "s2"]})
    pool = pool_sections([a, b], depth=3)
    assert pool["q0001"]["s2"] == ["a", "b"] and "s4" not in pool["q0001"]
    pairs = unlabelled_pairs([q1, q2], pool)
    assert [(p[0].qid, p[1]) for p in pairs] == [("q0001", "s2"), ("q0001", "s3")]

    calls: list[str] = []

    def judge(text: str, sid: str) -> tuple[int, str, str, str]:
        calls.append(sid)
        return (1 if sid == "s2" else 0, "r", "m", "v")

    js = judge_pairs(pairs, judge)
    path = tmp_path / "j.jsonl"
    write_judgments(path, js)
    done = read_judgments(path)
    assert all(not j.reviewed for j in done.values())
    again = judge_pairs(pairs, judge, done=done)  # cached: no new calls
    assert calls == ["s2", "s3"] and len(again) == 2

    v2 = build_v2([q1, q2], js)
    assert v2[0].relevant == {"s1": 2, "s2": 1}
    assert "v2: +1" in (v2[0].notes or "")
    assert v2[1] == q2
