"""BM25, BM25F and TF-IDF scores against values worked out by hand."""

import math
from pathlib import Path

import numpy as np
import pytest

from docseek.lexical.bm25 import BM25Scorer, lucene_idf
from docseek.lexical.inverted_index import InvertedIndex
from docseek.lexical.tfidf import TfidfScorer
from docseek.lexical.tokenizer import TokenizerConfig

RAW = TokenizerConfig(stopwords=False, stem=False)


def _single(texts: list[str]) -> InvertedIndex:
    return InvertedIndex.build(
        [f"d{i}" for i in range(len(texts))],
        [{"all": t} for t in texts],
        fields=("all",),
        tokenizer=RAW,
    )


def test_index_statistics() -> None:
    idx = _single(["cat dog", "cat cat mouse", "dog"])
    assert idx.n_docs == 3
    assert idx.df == {"cat": 2, "dog": 2, "mouse": 1}
    assert idx.lengths["all"].tolist() == [2, 3, 1]
    assert idx.avg_length("all") == 2.0
    p = idx.postings["all"]["cat"]
    assert p.docs.tolist() == [0, 1] and p.tfs.tolist() == [1, 2]


def test_bm25_matches_hand_computation() -> None:
    # N=3, avgdl=2, df(cat)=2 -> idf = ln(1 + 1.5/2.5) = ln 1.6
    # d0: tf=1, dl=2 -> B=1      -> 1*2.2/(1+1.2)                 = 1
    # d1: tf=2, dl=3 -> B=1.375  -> tf~=2/1.375=16/11 -> (16/11*2.2)/(1.2+16/11) = 3.2/2.6545...
    idx = _single(["cat dog", "cat cat mouse", "dog"])
    scores = BM25Scorer(idx, k1=1.2, b=0.75).score(["cat"])
    idf = math.log(1.6)
    assert lucene_idf(3, 2) == pytest.approx(idf)
    assert scores[0] == pytest.approx(idf * 1.0, rel=1e-6)
    assert scores[1] == pytest.approx(idf * 3.2 / (1.2 + 16 / 11), rel=1e-6)
    assert scores[1] == pytest.approx(0.56658, abs=1e-5)
    assert scores[2] == 0.0
    # Query terms are de-duplicated and unknown terms are ignored.
    assert np.allclose(BM25Scorer(idx).score(["cat", "cat", "zebra"]), scores)


def test_bm25_b0_ignores_length_and_k1_0_is_binary() -> None:
    idx = _single(["cat", "cat filler filler filler filler"])
    s = BM25Scorer(idx, k1=1.2, b=0.0).score(["cat"])
    assert s[0] == pytest.approx(s[1])
    idx2 = _single(["cat", "cat cat cat"])
    s2 = BM25Scorer(idx2, k1=0.0, b=0.0).score(["cat"])
    assert s2[0] == pytest.approx(s2[1]) == pytest.approx(lucene_idf(2, 2))


def test_bm25f_weights_fields_before_saturation() -> None:
    # b=0, title weight 2: d0 has cat in its title -> tf~=2; d1 only in body -> tf~=1.
    idx = InvertedIndex.build(
        ["d0", "d1"],
        [{"title": "cat", "body": "dog"}, {"title": "dog", "body": "cat"}],
        fields=("title", "body"),
        tokenizer=RAW,
    )
    s = BM25Scorer(idx, k1=1.2, b=0.0, field_weights={"title": 2.0, "body": 1.0}).score(["cat"])
    idf = math.log(1.2)  # N=2, df=2
    assert s[0] == pytest.approx(idf * 2 * 2.2 / 3.2, rel=1e-6)
    assert s[1] == pytest.approx(idf * 1.0, rel=1e-6)
    with pytest.raises(ValueError, match="unindexed"):
        BM25Scorer(idx, field_weights={"code": 1.0})


def test_tfidf_cosine_matches_hand_computation() -> None:
    # N=2; idf(cat)=ln(3/3)+1=1, idf(mouse)=idf(dog)=ln(3/2)+1.
    # d1 = [cat: 1+ln2, mouse: idf_m]; q = [mouse: idf_m] -> cos = idf_m / |d1|.
    idx = _single(["cat dog", "cat cat mouse"])
    s = TfidfScorer(idx).score(["mouse"])
    idf_m = math.log(1.5) + 1
    assert s[0] == 0.0
    assert s[1] == pytest.approx(idf_m / math.hypot(1 + math.log(2), idf_m), rel=1e-6)
    assert s[1] == pytest.approx(0.63871, abs=1e-5)
    assert TfidfScorer(idx).score(["zebra"]).sum() == 0.0


def test_index_round_trips_through_disk(tmp_path: Path) -> None:
    idx = _single(["cat dog", "cat cat mouse"])
    idx.save(tmp_path / "i.pkl")
    loaded = InvertedIndex.load(tmp_path / "i.pkl")
    assert loaded.doc_ids == idx.doc_ids and loaded.df == idx.df
    assert np.allclose(BM25Scorer(loaded).score(["cat"]), BM25Scorer(idx).score(["cat"]))
