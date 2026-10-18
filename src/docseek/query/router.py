"""Route a query to a pipeline by its predicted type.

The classifier only picks *which* retriever runs; every route enforces ACL itself, and an
unknown or failed classification falls back to ``default`` (degrade to hybrid, §7).
"""

from __future__ import annotations

from collections.abc import Mapping

from docseek.core.interfaces import Retriever
from docseek.core.models import Query, QueryType, ScoredHit
from docseek.core.tracing import Tracer
from docseek.query.classifier import QueryClassifier


class RouterRetriever:
    def __init__(
        self,
        name: str,
        classifier: QueryClassifier,
        routes: Mapping[QueryType, Retriever],
        default: Retriever,
        *,
        use_llm: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        self.name = name
        self._classifier = classifier
        self._routes = dict(routes)
        self._default = default
        self._use_llm = use_llm
        self._tracer = tracer
        self.decisions: dict[str, tuple[str, str, str]] = {}  # qid -> (qtype, source, route)

    def route(self, query: Query) -> tuple[Retriever, str, str]:
        c = self._classifier.classify(query.text, use_llm=self._use_llm)
        target = self._routes.get(c.qtype, self._default)
        return target, str(c.qtype), c.source

    def search(self, query: Query, k: int) -> list[ScoredHit]:
        target, qtype, source = self.route(query)
        self.decisions[query.qid] = (qtype, source, target.name)
        if self._tracer is not None:
            self._tracer.record(
                kind="understand",
                name="router",
                input={"qid": query.qid, "text": query.text},
                output={"qtype": qtype, "source": source, "route": target.name},
                latency_ms=0.0,
            )
        return target.search(query, k)
