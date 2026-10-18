"""Token counters."""

from pathlib import Path

import pytest

from docseek.core.config import load_models_config
from docseek.ingest.tokens import HFTokenCounter, RegexTokenCounter, TokenCounter


def test_regex_counter_counts_words_and_punctuation() -> None:
    counter = RegexTokenCounter()
    assert isinstance(counter, TokenCounter)
    assert counter.count("Use `Depends()`, please.") == 9  # Use ` Depends ( ) ` , please .
    assert counter.count("") == 0


@pytest.mark.network
def test_hf_counter_uses_wordpiece_of_default_embedder(configs_dir: Path) -> None:
    counter = HFTokenCounter(load_models_config(configs_dir).embedding.default)
    assert isinstance(counter, TokenCounter)
    # WordPiece splits identifiers ("BackgroundTasks" -> background ##tas ##ks).
    assert counter.count("BackgroundTasks") > RegexTokenCounter().count("BackgroundTasks")
    assert counter.count("") == 0
