"""CLIP image retriever on Qdrant ``:memory:`` with a fake encoder: ACL and named vector."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from docseek.core.acl import AccessPolicy
from docseek.core.config import UnknownPrincipalError, load_principals
from docseek.core.interfaces import FloatArray
from docseek.core.models import Chunk, Query
from docseek.dense.image_retriever import IMAGE_VECTOR, ImageRetriever, build_image_retriever
from docseek.ingest.images import IMAGE_STRATEGY
from docseek.testing.acl_checks import probe_leaks
from docseek.testing.fakes import HashEmbedder
from docseek.testing.fixtures import REPO_ROOT, mini_ingest

POLICY = AccessPolicy(load_principals(REPO_ROOT / "configs"))


class FakeClip:
    """'Pixels' are the file's text, embedded with the same hash space as the query."""

    name = "fake-clip"
    dim = 64

    def __init__(self) -> None:
        self._h = HashEmbedder(dim=self.dim)

    def embed_images(self, paths: Sequence[Path]) -> FloatArray:
        return self._h.embed_documents([p.read_text(encoding="utf-8") for p in paths])

    def embed_query(self, text: str) -> FloatArray:
        return self._h.embed_query(text)


def _image_chunks(root: Path) -> list[Chunk]:
    out = []
    for i, c in enumerate(mini_ingest().chunks):
        rel = f"img/{i}.png"
        (root / "img").mkdir(exist_ok=True)
        (root / rel).write_text(c.text, encoding="utf-8")
        out.append(
            Chunk(
                chunk_id=f"{c.section_id}::{IMAGE_STRATEGY}::{i}",
                section_id=c.section_id,
                strategy=IMAGE_STRATEGY,
                text="",
                text_for_embedding="",
                metadata={**c.metadata, "modality": "image", "image_path": rel},
                n_tokens=0,
            )
        )
    return out


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[ImageRetriever, list[Chunk]]:
    root = tmp_path_factory.mktemp("imgs")
    images = _image_chunks(root)
    text_chunks = mini_ingest().chunks  # ignored by the image retriever
    retriever = build_image_retriever(
        {"location": ":memory:", "repo_root": str(root), "cache_dir": None},
        [*text_chunks, *images],
        POLICY,
        encoder=FakeClip(),
    )
    return retriever, images


def test_no_leakage_for_any_principal(setup: tuple[ImageRetriever, list[Chunk]]) -> None:
    retriever, images = setup
    acl = {c.chunk_id: c.metadata["acl"] for c in images}
    result = probe_leaks(retriever, acl, POLICY)
    assert result.leaks == []
    assert result.restricted_seen["anonymous"] == 0
    assert result.restricted_seen["engineer"] > 0


def test_only_image_chunks_on_named_vector(setup: tuple[ImageRetriever, list[Chunk]]) -> None:
    retriever, images = setup
    hits = retriever.search(Query(qid="q", text="BackgroundTasks", principal="security"), 50)
    assert hits and all(h.chunk_id.split("::")[1] == IMAGE_STRATEGY for h in hits)
    assert retriever._store.vector_name == IMAGE_VECTOR
    assert retriever._store.count() == len(images)


def test_unknown_principal_fails_closed(setup: tuple[ImageRetriever, list[Chunk]]) -> None:
    with pytest.raises(UnknownPrincipalError):
        setup[0].search(Query(qid="x", text="payment", principal="intruder"), 5)


def test_requires_image_chunks() -> None:
    with pytest.raises(ValueError, match="image chunks"):
        build_image_retriever(
            {"location": ":memory:"}, mini_ingest().chunks, POLICY, encoder=FakeClip()
        )
