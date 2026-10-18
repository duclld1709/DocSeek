"""Write the M5 chunking variants next to the section chunks:

- data/processed/chunks_fixed_256.jsonl  (256-token document windows, 32 overlap, bge tokenizer)
- data/processed/chunks_section_ctx.jsonl (section chunks + path header in text_for_embedding)

    uv run python scripts/m5_chunk_variants.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docseek.core.config import load_models_config
from docseek.core.models import Chunk, Section
from docseek.ingest.pipeline import read_jsonl, write_jsonl
from docseek.ingest.strategies import fixed_chunks, hf_offsets, with_context_header


def main() -> None:
    root = Path("data/processed")
    sections = read_jsonl(root / "sections.jsonl", Section)
    base = read_jsonl(root / "chunks.jsonl", Chunk)
    model = load_models_config().embedding.default
    fixed = fixed_chunks(sections, hf_offsets(model), size=256, overlap=32)
    ctx = with_context_header(base, sections)
    stats: dict[str, dict[str, int]] = {}
    for name, chunks in (("fixed_256", fixed), ("section_ctx", ctx)):
        n = write_jsonl(root / f"chunks_{name}.jsonl", chunks)
        stats[name] = {"chunks": n, "sections_covered": len({c.section_id for c in chunks})}
    stats["section"] = {"chunks": len(base), "sections_covered": len({c.section_id for c in base})}
    (root / "chunk_variants.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
