"""Cartesian parameter sweeps over an experiment config (``docseek eval grid``).

Each grid point is an ordinary run directory; the sweep adds
``experiments/<ts>_<name>_grid_<split>/summary.json`` listing every point with its run
directory and headline metrics, for plotting and for picking the best configuration.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from docseek.core.runs import make_run_dir, write_json
from docseek.eval.runner import ExperimentConfig


def parse_param(spec: str) -> tuple[str, list[Any]]:
    """``"retriever.k1=0.9,1.2"`` → ``("retriever.k1", [0.9, 1.2])`` (values parsed as YAML)."""
    key, sep, values = spec.partition("=")
    if not sep or not key or not values:
        raise ValueError(f"expected key=v1,v2,... got {spec!r}")
    return key.strip(), [yaml.safe_load(v) for v in values.split(",")]


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(value).lower()).strip("-") or "x"


def expand(
    base: ExperimentConfig, params: Mapping[str, Sequence[Any]]
) -> list[tuple[dict[str, Any], ExperimentConfig]]:
    """Every combination of ``params`` applied to ``base``, with a descriptive name each."""
    keys = list(params)
    out: list[tuple[dict[str, Any], ExperimentConfig]] = []
    for combo in itertools.product(*(params[k] for k in keys)):
        data = base.model_dump(mode="json")
        point = dict(zip(keys, combo, strict=True))
        for k, v in point.items():
            _set(data, k, v)
        suffix = "_".join(f"{k.split('.')[-1].lower()}-{_slug(v)}" for k, v in point.items())
        data["name"] = f"{base.name}_{suffix}"
        out.append((point, ExperimentConfig.model_validate(data)))
    return out


def run_grid(
    base: ExperimentConfig,
    params: Mapping[str, Sequence[Any]],
    run_one: Callable[[ExperimentConfig], Path],
    *,
    split: str,
    experiments_dir: Path,
    metrics: Sequence[str] = ("ndcg@10", "recall@10", "mrr@10"),
) -> Path:
    """Run every grid point with ``run_one`` and write the sweep summary; returns its dir."""
    rows: list[dict[str, Any]] = []
    for point, cfg in expand(base, params):
        run_dir = run_one(cfg)
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["overall"]
        rows.append(
            {
                "params": point,
                "run_dir": run_dir.as_posix(),
                **{name: m[name]["mean"] for name in metrics if name in m},
            }
        )
    out = make_run_dir(experiments_dir, f"{base.name}_grid_{split}")
    write_json(
        out / "summary.json",
        {"base": base.name, "split": split, "params": dict(params), "points": rows},
    )
    return out
