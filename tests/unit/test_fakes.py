import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from docseek.core.interfaces import Embedder, LLMClient, Reranker
from docseek.core.models import LLMResponse, ScoredHit, ToolCall
from docseek.testing.fakes import HashEmbedder, IdentityReranker, ScriptedLLM, ScriptExhaustedError


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(HashEmbedder(), Embedder)
    assert isinstance(IdentityReranker(), Reranker)
    assert isinstance(ScriptedLLM(replies=[]), LLMClient)


EXPECTED_FASTAPI_DIM8 = [0.4714, 0.0, -0.7071, 0.0, -0.2357, -0.4714, 0.0, 0.0]


class TestHashEmbedder:
    def test_shapes_dtype_and_unit_norm(self) -> None:
        emb = HashEmbedder(dim=32)
        docs = emb.embed_documents(["BackgroundTasks", "Depends with yield"])

        assert docs.shape == (2, 32)
        assert docs.dtype == np.float32
        assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-6)
        assert emb.embed_query("x y z").shape == (32,)

    def test_is_deterministic_and_matches_known_value(self) -> None:
        # Pinned value guards against accidental use of the per-process salted hash().
        vec = HashEmbedder(dim=8).embed_query("fastapi")
        assert np.array_equal(vec, HashEmbedder(dim=8).embed_query("fastapi"))
        assert vec.round(4).tolist() == pytest.approx(EXPECTED_FASTAPI_DIM8, abs=1e-4)

    def test_similar_texts_score_higher(self) -> None:
        emb = HashEmbedder(dim=256)
        q = emb.embed_query("dependency with yield")
        near, far = emb.embed_documents(["dependencies with yield", "upload multiple files"])
        assert float(q @ near) > float(q @ far)

    def test_normalises_case_and_whitespace(self) -> None:
        emb = HashEmbedder()
        assert np.array_equal(emb.embed_query("Path  Params"), emb.embed_query("path params"))

    def test_empty_inputs(self) -> None:
        emb = HashEmbedder(dim=16)
        assert emb.embed_documents([]).shape == (0, 16)
        assert not emb.embed_query("").any()

    @pytest.mark.parametrize(("dim", "ngrams"), [(0, (3, 5)), (8, (0, 2)), (8, (4, 3))])
    def test_rejects_bad_config(self, dim: int, ngrams: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            HashEmbedder(dim=dim, ngram_range=ngrams)


def _hit(i: int) -> ScoredHit:
    return ScoredHit(
        chunk_id=f"c{i}", section_id=f"d:x.md#s{i}", score=1 / i, rank=i, retriever="bm25"
    )


def test_identity_reranker_keeps_order_and_truncates() -> None:
    hits = [_hit(i) for i in (1, 2, 3)]
    out = IdentityReranker().rerank("q", hits, texts={}, k=2)

    assert [h.chunk_id for h in out] == ["c1", "c2"]
    assert [h.rank for h in out] == [1, 2]
    assert out[0].retriever == "bm25+identity"
    assert hits[0].retriever == "bm25"  # inputs are not mutated


class Verdict(BaseModel):
    score: int
    reason: str


class TestScriptedLLM:
    def test_replays_in_order_and_records_calls(self) -> None:
        llm = ScriptedLLM(replies=["first", LLMResponse(content="second", model="m", tokens_in=3)])

        assert llm.complete([{"role": "user", "content": "a"}]).content == "first"
        second = llm.complete([{"role": "user", "content": "b"}], model="other", temperature=0.7)
        assert second.tokens_in == 3
        assert [c["messages"][0]["content"] for c in llm.calls] == ["a", "b"]
        assert llm.calls[1]["temperature"] == 0.7

    def test_raises_when_script_exhausted(self) -> None:
        llm = ScriptedLLM(replies=["only"])
        llm.complete([])
        with pytest.raises(ScriptExhaustedError):
            llm.complete([])

    def test_structured_output_is_validated(self) -> None:
        llm = ScriptedLLM(replies=['{"score": 2, "reason": "direct"}', '{"score": "high"}'])

        ok = llm.complete([], response_format=Verdict)
        assert ok.parsed == {"score": 2, "reason": "direct"}
        with pytest.raises(ValidationError):
            llm.complete([], response_format=Verdict)

    def test_tool_calls_pass_through(self) -> None:
        call = ToolCall(id="1", name="search", arguments={"query": "cors"})
        llm = ScriptedLLM(replies=[LLMResponse(model="m", tool_calls=[call])])
        tools = [{"type": "function", "function": {"name": "search"}}]

        out = llm.complete([], tools=tools)
        assert out.tool_calls == [call]
        assert llm.calls[0]["tools"] == tools
