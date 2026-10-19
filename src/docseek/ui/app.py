"""DocSeek demo UI: Compare / Answer / Trace / Eval.

    uv run streamlit run src/docseek/ui/app.py      # API at $DOCSEEK_API_URL (default :8000)

The principal picker only sets the ``X-Principal`` header of the demo API (simulated auth);
ACL is enforced by the API's retrievers and tools, never by this UI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API = os.environ.get("DOCSEEK_API_URL", "http://127.0.0.1:8000")
EXPERIMENTS = Path(os.environ.get("DOCSEEK_EXPERIMENTS_DIR", "experiments"))
PRINCIPALS = ["anonymous", "engineer", "security"]
SEARCH_MODES = ["bm25", "dense", "hybrid", "rerank"]


def call(method: str, path: str, principal: str, **kw: Any) -> dict[str, Any]:
    r = httpx.request(method, f"{API}{path}", headers={"X-Principal": principal}, timeout=180, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    data: dict[str, Any] = r.json()
    return data


def show_source(section_id: str, principal: str) -> None:
    try:
        sec = call("GET", f"/sections/{quote(section_id, safe='')}", principal)
    except RuntimeError as exc:
        st.warning(str(exc))
        return
    st.caption(f"{sec.get('title')} · `{section_id}`")
    if sec.get("file") and sec.get("page"):
        st.caption(f"Scanned PDF: page {sec['page']} of {sec['file']}")
        pdf = Path(sec["file"])
        if pdf.is_file():
            import pymupdf

            with pymupdf.open(str(pdf)) as doc:
                st.image(doc[int(sec["page"]) - 1].get_pixmap(dpi=80).tobytes("png"))
    if sec.get("image_path"):
        img = Path("data/raw/fastapi") / sec["image_path"]
        if img.is_file():
            st.image(str(img))
    st.text(sec.get("text", "")[:3000])


def tab_compare(query: str, principal: str, k: int) -> None:
    if not query:
        st.info("Type a query above.")
        return
    cols = st.columns(len(SEARCH_MODES))
    for col, mode in zip(cols, SEARCH_MODES, strict=True):
        with col:
            st.subheader(mode)
            try:
                res = call(
                    "POST", "/search", principal, json={"query": query, "mode": mode, "k": k}
                )
            except RuntimeError as exc:
                st.error(str(exc))
                continue
            st.caption(f"{res['latency_ms']:.0f} ms · trace `{res['trace_id'][:8]}`")
            for h in res["hits"]:
                badge = "" if h["modality"] == "text" else f" [{h['modality']}]"
                st.markdown(f"**{h['rank']}. {h['title']}**{badge}  \n`{h['section_id']}`")
                st.caption(h["snippet"][:200])


def tab_answer(query: str, principal: str, mode: str) -> None:
    if not query:
        st.info("Type a question above.")
        return
    if st.button("Ask", type="primary"):
        with st.spinner("Answering..."):
            try:
                st.session_state["answer"] = call(
                    "POST", "/ask", principal, json={"query": query, "mode": mode, "debug": True}
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return
    ans = st.session_state.get("answer")
    if not ans:
        return
    st.session_state["trace_id"] = ans.get("trace_id")
    st.markdown(ans["answer"])
    meta = (
        f"mode {ans['mode']} · {ans['steps']} step(s) · {ans['latency_ms'] / 1000:.1f} s · "
        f"stop: {ans['stop_reason']} · trace `{(ans.get('trace_id') or '')[:8]}`"
    )
    st.caption(meta)
    if ans["removed_citations"]:
        st.warning(f"Removed citations not in the retrieved evidence: {ans['removed_citations']}")
    if ans["abstained"] and ans["suggestions"]:
        st.write("Closest sections:")
        for s in ans["suggestions"]:
            st.markdown(f"- {s['title']} `{s['section_id']}`")
    for sid in ans["citations"]:
        with st.expander(f"Source: {sid}"):
            show_source(sid, principal)


def tab_trace(principal: str) -> None:
    tid = st.text_input("Trace id", value=st.session_state.get("trace_id") or "")
    if not tid:
        st.info("Ask a question first, or paste a trace id.")
        return
    try:
        tr = call("GET", f"/traces/{tid}", principal)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"{len(tr['events'])} events · {tr['total_latency_ms']:.0f} ms · "
        f"${tr['total_cost_usd']:.5f}"
    )
    rows = [
        {
            "step": e["step"],
            "kind": e["kind"],
            "name": e["name"],
            "latency_ms": round(e["latency_ms"], 1),
            "tokens_in": e["tokens_in"],
            "tokens_out": e["tokens_out"],
            "cost_usd": e["cost_usd"],
            "detail": json.dumps(e["input"].get("args") or e["output"], ensure_ascii=False)[:160],
        }
        for e in tr["events"]
    ]
    st.bar_chart({r["step"]: r["latency_ms"] for r in rows}, x_label="step", y_label="ms")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _mean(overall: dict[str, Any], key: str) -> float | None:
    value = overall.get(key)
    return value.get("mean") if isinstance(value, dict) else value


def tab_eval() -> None:
    runs = sorted(EXPERIMENTS.glob("*_test/metrics.json"))
    if not runs:
        st.info(f"No test runs under {EXPERIMENTS}/")
        return
    rows = []
    for path in runs:
        m = json.loads(path.read_text(encoding="utf-8"))
        overall = m.get("overall", {})
        rows.append(
            {
                "run": path.parent.name,
                "config": m.get("name"),
                "nDCG@10": _mean(overall, "ndcg@10"),
                "Recall@10": _mean(overall, "recall@10"),
                "MRR@10": _mean(overall, "mrr@10"),
                "n": overall.get("n_queries"),
                "p95 ms": (m.get("latency_ms") or {}).get("p95"),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    chosen = [r for r in rows if r["nDCG@10"] is not None][-15:]
    st.bar_chart(
        {f"{r['config']} ({r['run'][:15]})": r["nDCG@10"] for r in chosen}, horizontal=True
    )


def main() -> None:
    st.set_page_config(page_title="DocSeek", layout="wide")
    st.title("DocSeek")
    st.caption("Agentic retrieval over the FastAPI docs + an internal KB. Demo auth: X-Principal.")
    c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
    params = st.query_params  # ?q=...&principal=... prefill (shareable links, screenshots)
    query = c1.text_input(
        "Query",
        value=params.get("q", ""),
        placeholder="How do I run a task after returning a response?",
    )
    start = params.get("principal", "anonymous")
    principal = c2.selectbox(
        "Principal", PRINCIPALS, index=PRINCIPALS.index(start) if start in PRINCIPALS else 0
    )
    mode = c3.selectbox("Answer mode", ["agentic", "one-shot", "agentic-no-fs"])
    k = int(c4.number_input("k", 1, 20, 5))
    compare, answer, trace, evals = st.tabs(["Compare", "Answer", "Trace", "Eval"])
    with compare:
        tab_compare(query, principal, k)
    with answer:
        tab_answer(query, principal, mode)
    with trace:
        tab_trace(principal)
    with evals:
        tab_eval()


main()
