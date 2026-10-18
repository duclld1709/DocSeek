"""fastembed wrapper (downloads the model once; marker `network`)."""

from pathlib import Path

import numpy as np
import pytest

from docseek.core.config import load_models_config
from docseek.core.interfaces import Embedder
from docseek.dense.embedders import FastEmbedEmbedder


@pytest.mark.network
def test_fastembed_embedder_shapes_norms_and_similarity(configs_dir: Path) -> None:
    emb = FastEmbedEmbedder(load_models_config(configs_dir).embedding.default)
    assert isinstance(emb, Embedder) and emb.dim == 384
    docs = emb.embed_documents(["how to use Depends", "how do I use Depends()?", "docker"])
    assert docs.shape == (3, 384) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    sims = docs @ docs.T
    assert sims[0, 1] > 0.9 > sims[0, 2]
    assert emb.embed_query("how to use Depends").shape == (384,)
    assert emb.embed_documents([]).shape == (0, 384)
