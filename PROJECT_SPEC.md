# PROJECT_SPEC.md — DocSeek

> **Measurable Agentic Retrieval over Technical Documentation**
> Capstone project bám theo syllabus "Production-Grade Agentic RAG" (8 buổi, 2 act).
> Spec này viết cho **Claude Code** đọc và thực hiện. Đồng thời nó cũng là **đề bài** để người học tự code lại rồi so sánh.

---

## 0. Cách dùng spec này

### 0.1. Hai mục đích của dự án

1. **Dự án mẫu để học.** Mỗi milestone có đề bài riêng, acceptance criteria và một git tag. Người học tự code milestone đó trước, sau đó checkout tag của bản tham chiếu, chạy **cùng một bộ eval** rồi so sánh.
2. **Dự án đưa vào CV.** Repo phải có README chuẩn, sơ đồ kiến trúc, bảng benchmark với **số đo thật**, CI xanh, Docker chạy được bằng một lệnh, và có ADR (Architecture Decision Records) giải thích các tradeoff.

### 0.2. Quy trình làm việc với Claude Code (VS Code)

- **Mỗi milestone một phiên làm việc.** Mở đầu phiên bằng prompt mẫu ở Phụ lục A.
- Claude Code **phải đọc lại** `PROJECT_SPEC.md`, `CLAUDE.md` và `docs/milestones/Mx.md` trước khi code.
- Khi xong một milestone, thực hiện theo thứ tự:
  1. Chạy `docseek eval` (khi milestone có eval).
  2. Cập nhật `docs/results/Mx.md`.
  3. Commit.
  4. Tạo tag `mx-<slug>`, ví dụ `m2-bm25`.
  5. Push.
- Không gộp nhiều milestone vào một commit lớn. Commit nhỏ, message theo Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`, `refactor:`).

### 0.3. Quy trình đề xuất cho người học (tự code rồi so sánh)

```
1. Đọc docs/milestones/Mx.md (đề bài và acceptance criteria)
2. git checkout m(x-1)-...  →  git switch -c my/mx
3. Tự code. Chỉ được dùng eval harness và golden set của bản tham chiếu (đó là "giám khảo" chung)
4. docseek eval --config configs/my_mx.yaml --tag my-mx
5. git diff my/mx m{x}-...  +  so sánh docs/results/Mx.md với kết quả của mình
6. Ghi vào docs/learning-log/Mx.md: mình làm khác gì, kết quả chênh bao nhiêu, vì sao
```

> Quy tắc vàng: **cùng golden set, cùng split, cùng metric**. Không đúng ba điều này thì so sánh không có ý nghĩa.

---

## 1. Bối cảnh và mục tiêu

### 1.1. Vấn đề

Nhiều hệ thống RAG thất bại không phải vì model yếu, mà vì retrieval không biết ưu tiên relevance, metadata, freshness, permissions, cấu trúc tài liệu và ý định của query. Hệ quả thường gặp:

- Không có benchmark, nên không biết hệ thống đang tốt lên hay tệ đi.
- Không phân biệt được lỗi đến từ dữ liệu, từ retrieval hay từ model.

### 1.2. Mục tiêu sản phẩm

Xây **DocSeek**, một hệ thống search và trả lời câu hỏi trên tài liệu kỹ thuật (FastAPI docs, cộng các phần mở rộng), có các đặc điểm:

- **Search stack đầy đủ:** BM25 tự cài → dense (Qdrant) → hybrid (RRF/DBSF) → cross-encoder rerank.
- **Agentic retrieval:** query understanding (rewrite, decomposition, metadata extraction), router, knowledge graph (Neo4j), và filesystem tools (`ls`/`grep`/`cat`/`find`).
- **Ingestion có trace về nguồn:** markdown parsing, OCR cho PDF scan, bảng, ảnh chụp màn hình (multimodal).
- **Đo lường được:** golden set, metric IR tự cài và kiểm chứng được, DeepEval, LLM-as-judge đã hiệu chỉnh, kiểm định thống kê, regression gate.
- **Production tradeoffs:** báo cáo quality, latency (p50/p95), RAM, disk, thời gian build index, và cost cho mỗi 1.000 query.

### 1.3. Mục tiêu học tập (map với syllabus)

| Buổi | Chủ đề | Milestone |
|---|---|---|
| 01 | Modern search overview, dataset, project | M0 |
| 02 | Evaluation: synthetic data, judges, metrics | M1 |
| 03 | Lexical: TF-IDF → BM25 | M2 |
| 04 | Semantic search với Qdrant | M3 |
| 05 | Hybrid + reranking | M4 |
| 06 | Query understanding, orchestration, metadata, chunking | M5 |
| 07 | Knowledge graph + filesystem search | M6 |
| 08 | Document parsing + multimodal | M7 |
| Final | Agent, API, UI, benchmark report, CV | M8 |

### 1.4. Non-goals

- Không fine-tune embedding model (có thể làm như phần mở rộng sau M8).
- Không xây UI production. Streamlit đủ cho demo.
- Không dùng framework "đóng hộp" che hết logic như LangChain RetrievalQA hay LlamaIndex query engine. Mục tiêu là **hiểu từng tầng**, nên agent loop, fusion, metric và BM25 đều tự cài. Thư viện chỉ dùng để **đối chiếu**.
- Không deploy lên cloud. Docker compose chạy local là đủ. Khoá "Production AI" sau này sẽ lo phần deployment.

---

## 2. Quyết định đã chốt

| Hạng mục | Quyết định |
|---|---|
| Tên dự án / package | `docseek` |
| Corpus chính | FastAPI docs tiếng Anh (`docs/en/docs` + `docs_src`), **pin commit** `b23d78f2ce4e7b3ea48a59f51ca249690d9fd02c` (2026-10-17) |
| Corpus mở rộng | (a) Internal KB giả lập có ACL và freshness; (b) tập PDF scan do mình render; (c) ảnh trong docs (multimodal) |
| Embedding / sparse / rerank | Chạy **local** (sentence-transformers / fastembed), CPU là đủ |
| LLM | Gọi qua **LiteLLM**, provider-agnostic. Có sẵn key **Gemini** và **OpenAI** |
| Phân vai LLM | Generator (sinh query, caption, rewrite, agent) dùng model rẻ. Judge dùng model **khác họ** với generator để giảm self-preference bias. Mặc định: generate bằng Gemini, judge bằng OpenAI. Đổi được qua config |
| Vector DB | Qdrant (docker; test dùng `QdrantClient(":memory:")`) |
| Graph DB | Neo4j Community (docker) |
| Eval | Metric tự cài, kiểm chứng chéo với `ranx`; DeepEval cho LLM-judge và metric phía answer |
| Python | 3.12, quản lý bằng **uv** |
| Giao nhận | GitHub repo của người học; mỗi milestone có tag; GitHub Actions CI |
| Ngôn ngữ | Code, docstring, README, ADR viết **tiếng Anh**. `docs/learning-log/` và `docs/milestones/` được phép viết tiếng Việt |

---

## 3. Dữ liệu

### 3.1. Corpus A: FastAPI docs (public)

- **Nguồn:** `https://github.com/fastapi/fastapi`, sparse checkout `docs/en/docs` và `docs_src` tại commit đã pin. Khoảng 156 file `.md` và khoảng 464 file `.py` ví dụ.
- **Script:** `docseek data fetch` clone sparse vào `data/raw/fastapi/` và ghi `data/raw/MANIFEST.json` gồm commit, thời điểm, số file và sha256 từng file.
- **Loại trừ:** `release-notes.md` không index như tài liệu thường (chỉ dùng cho KG/version ở M6). Cũng loại `fastapi-people.md`, `external-links.md`, `_llm-test.md`, `translation-banner.md`, `newsletter.md`, cùng thư mục `css/` và `js/`. Danh sách cụ thể đặt trong `configs/corpus.yaml`.
- **Những điểm parse đặc thù, cần xử lý (đây là bài học về parsing):**
  - Heading có anchor: `## First Steps { #first-steps }`. Tách thành `title="First Steps"` và `anchor="first-steps"`.
  - Code include: `{* ../../docs_src/dependencies/tutorial001_an_py310.py hl[8:9] *}`. Phải **resolve** thành nội dung file `.py` thật. Nếu có nhiều biến thể (`_py39`, `_py310`, `_an_`), ưu tiên `_an_py310`. Ghi `code_refs` vào metadata.
  - HTML inline như `<dfn title="...">`, admonition `/// tip | ... ///`, tab và bảng markdown.
  - Link nội bộ `[text](../path.md#anchor)`. Giữ lại để dựng graph `LINKS_TO`.
- **Metadata suy ra:**
  - `doc_type` theo thư mục: `tutorial | advanced | how-to | reference | deployment | learn | about | other`.
  - `section_path` lấy từ chuỗi heading, ví dụ `Tutorial > Dependencies > First Steps`.
  - `updated_at` lấy từ `git log -1 --format=%cI -- <file>`. Muốn có giá trị này thì clone đủ history của các file docs, hoặc dùng GitHub API.
  - `acl = ["public"]`, `source="fastapi-docs"`.

### 3.2. Corpus B: Internal KB giả lập (permissions + freshness)

Mục đích là tạo bài toán "Enterprise KB" có thật trên cùng hệ thống.

- **Bối cảnh hư cấu:** công ty **"Acme Payments"** có các service viết bằng FastAPI.
- Sinh khoảng **40–60 tài liệu** bằng LLM (`docseek data gen-internal`), commit kết quả vào `data/internal_kb/` để tái lập được (không sinh lại mỗi lần chạy). Gồm các loại: runbook, ADR nội bộ, postmortem, onboarding guide, API convention, security policy.
- Mỗi doc có front-matter YAML:
  ```yaml
  id: kb-0012
  title: "Rotating JWT signing keys"
  acl: ["eng-internal"]          # public | eng-internal | security-team
  version: 3
  updated_at: 2026-08-02
  supersedes: kb-0007            # doc cũ hơn cùng chủ đề (có thể null)
  status: active                 # active | deprecated
  owner: platform-team
  ```
- **Cố ý tạo tình huống khó:**
  - Khoảng 8 cặp tài liệu **cũ/mới mâu thuẫn nhau**, ví dụ timeout 30s trong bản cũ và 10s trong bản mới.
  - Khoảng 6 doc `security-team` chứa thông tin mà user `eng-internal` **không được thấy**.
  - Vài doc tham chiếu khái niệm FastAPI, ví dụ "dùng `Depends` cho auth". Những tham chiếu này tạo cạnh nối giữa hai corpus trong graph.
- **Principal giả lập** (định nghĩa trong `configs/principals.yaml`):
  - `anonymous` → `[public]`
  - `engineer` → `[public, eng-internal]`
  - `security` → `[public, eng-internal, security-team]`

### 3.3. Corpus C: PDF scan (OCR + layout)

- Chọn khoảng **20 trang** FastAPI docs, trong đó có trang chứa bảng và code block. Thêm **khoảng 10 doc** từ Internal KB, gồm một bảng "SLA by service" và một bảng "Error codes".
- **Pipeline sinh dữ liệu** (`docseek data make-scans`):
  1. Render markdown sang HTML rồi sang PDF (weasyprint hoặc playwright print-to-pdf).
  2. Rasterize ở 150–200 DPI.
  3. Thêm nhiễu kiểu scan: xoay ±2°, blur nhẹ, noise, giảm contrast. Một số trang đặt 2 cột.
  4. Ghép lại thành PDF **không có text layer**.
- **Ground truth:** giữ text gốc trong `data/scans/ground_truth/*.txt`. Nhờ vậy có thể **đo CER/WER của OCR**. Đây là điểm rất mạnh để đưa vào báo cáo.
- `acl` và metadata được giữ nguyên theo tài liệu gốc. `modality="scan"`.

### 3.4. Corpus D: Ảnh (multimodal)

- Lấy các ảnh trong `docs/en/docs/img/`: screenshot Swagger UI, ReDoc, sơ đồ. Lọc ảnh dưới 5KB và icon. Giới hạn khoảng 150 ảnh.
- Mỗi ảnh có các trường:
  - `alt_text` và đoạn văn xung quanh nơi ảnh được chèn trong markdown.
  - `ocr_text`.
  - `caption` do vision LLM sinh, có cache.
  - CLIP image embedding.

### 3.5. Đơn vị dữ liệu và ID ổn định

```
Document  : doc_id = "<source>:<relative_path>"          vd "fastapi-docs:tutorial/dependencies/index.md"
Section   : section_id = "<doc_id>#<anchor>"             đơn vị CHẤM RELEVANCE (bất biến với chunking)
Chunk     : chunk_id = "<section_id>::<strategy>::<n>"   đơn vị INDEX (thay đổi theo chunking strategy)
```

> **Quyết định quan trọng:** golden set gán nhãn ở **mức section**, không ở mức chunk. Khi đánh giá, kết quả chunk được map về section (giữ rank tốt nhất của mỗi section, bỏ trùng). Nhờ vậy có thể so sánh công bằng các chunking strategy khác nhau trên cùng golden set. Ghi quyết định này vào `docs/adr/0002-section-level-relevance.md`.

---

## 4. Kiến trúc

### 4.1. Sơ đồ tổng thể

```
                ┌──────────────────────── INGESTION ────────────────────────┐
 raw sources →  │ fetch → parse (md / pdf+OCR / image) → normalize → chunk   │ → data/processed/*.jsonl
                │        → metadata extract → markdown projection (FS view)  │ → data/projected/**.md
                └────────────────────────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼───────────────────────────┬─────────────────────┐
          ▼                          ▼                           ▼                     ▼
   BM25 index (own)          Qdrant collection            Neo4j graph           Filesystem view
   data/index/bm25/          named vectors: dense,        Page/Section/Concept/  ls/grep/cat/find
                             sparse, image + payload      APIElement/Version     (sandboxed)
          └──────────────┬───────────┴──────────────┬────────────┘                     │
                         ▼                          ▼                                  │
                 Retrievers (Protocol)  →  Fusion (RRF/DBSF/weighted) → Reranker       │
                         ▲                                                             │
                         │                                                             │
   Query ─→ QueryUnderstanding (classify, rewrite, decompose, extract filters, ACL) ─→ Router
                         │
                         ▼
                Agent loop (tool calling via LiteLLM) ─→ Answer + citations + Trace (JSONL)
                         │
                FastAPI service  ·  Streamlit UI  ·  CLI (Typer)
                         │
                Eval harness (golden set, metrics, DeepEval, judge, bootstrap CI, regression gate)
```

### 4.2. Data models (pydantic v2, đặt ở `docseek/core/models.py`)

```python
class Section(BaseModel):
    section_id: str; doc_id: str; source: str
    title: str; anchor: str; section_path: list[str]
    text: str                      # đã resolve code include
    code_refs: list[str] = []
    links: list[str] = []          # section_id/doc_id đích
    doc_type: str; modality: Literal["text", "scan", "image"] = "text"
    acl: list[str]; updated_at: datetime | None
    version: int | None = None; status: str = "active"; supersedes: str | None = None
    provenance: Provenance         # file, line_start, line_end, page (PDF), bbox (optional)

class Chunk(BaseModel):
    chunk_id: str; section_id: str; strategy: str
    text: str; text_for_embedding: str   # có thể chèn header ngữ cảnh
    metadata: dict[str, Any]             # payload phẳng cho Qdrant filter
    n_tokens: int

class Query(BaseModel):
    qid: str; text: str
    principal: str = "anonymous"
    filters: MetadataFilter | None = None
    qtype: QueryType | None = None       # factoid|howto|conceptual|symbol|multihop|freshness|permission|scan|image|unanswerable

class ScoredHit(BaseModel):
    chunk_id: str; section_id: str; score: float; rank: int
    retriever: str; debug: dict[str, Any] = {}

class RetrievalResult(BaseModel):
    query: Query; hits: list[ScoredHit]; latency_ms: float; trace_id: str

class TraceEvent(BaseModel):
    trace_id: str; step: int; kind: str  # understand|retrieve|fuse|rerank|tool_call|llm_call|answer
    name: str; input: dict; output: dict; latency_ms: float
    tokens_in: int = 0; tokens_out: int = 0; cost_usd: float = 0.0
```

### 4.3. Interfaces (`typing.Protocol`, đặt ở `docseek/core/interfaces.py`)

```python
class Retriever(Protocol):
    name: str
    def search(self, query: Query, k: int) -> list[ScoredHit]: ...

class Embedder(Protocol):
    name: str; dim: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: str, hits: list[ScoredHit], texts: dict[str, str], k: int) -> list[ScoredHit]: ...

class LLMClient(Protocol):
    def complete(self, messages: list[dict], *, model: str | None = None,
                 tools: list[dict] | None = None, response_format: type[BaseModel] | None = None,
                 temperature: float = 0.0) -> LLMResponse: ...
```

**Yêu cầu bắt buộc:**

- Mỗi interface có một bản **Fake** trong `docseek/testing/fakes.py`:
  - `HashEmbedder`: embedding tất định từ hash n-gram.
  - `IdentityReranker`.
  - `ScriptedLLM`: trả lời theo kịch bản cho trước.
- Toàn bộ unit test và CI chạy bằng fake, **không cần mạng, không cần key**.
- **Mọi LLM call đi qua `CachedLLMClient`** (SQLite/diskcache, key là `hash(model, messages, tools, temperature)`). Chạy lại thì không tốn tiền và cho kết quả tất định. Cost được tính bằng `litellm.completion_cost` và ghi vào trace.

### 4.4. Cấu trúc repo

```
docseek/
├── PROJECT_SPEC.md            ← file này
├── CLAUDE.md                  ← quy tắc ngắn cho Claude Code (xem Phụ lục B)
├── README.md                  ← viết cho nhà tuyển dụng (xem §9)
├── pyproject.toml  uv.lock  .python-version  .env.example  Makefile
├── docker-compose.yml         ← qdrant, neo4j, api, ui
├── Dockerfile
├── .github/workflows/ci.yml
├── configs/
│   ├── corpus.yaml  principals.yaml  models.yaml
│   └── experiments/           ← mỗi cấu hình retrieval là 1 file yaml (bm25.yaml, dense_bge_small.yaml, hybrid_rrf.yaml, ...)
├── src/docseek/
│   ├── core/          models.py interfaces.py config.py logging.py tracing.py
│   ├── data/          fetch.py gen_internal.py make_scans.py
│   ├── ingest/        markdown_parser.py pdf_parser.py ocr.py image_ingest.py chunking.py metadata.py projection.py pipeline.py
│   ├── lexical/       tokenizer.py inverted_index.py bm25.py tfidf.py
│   ├── dense/         embedders.py qdrant_store.py
│   ├── hybrid/        fusion.py normalization.py
│   ├── rerank/        cross_encoder.py llm_reranker.py
│   ├── query/         classifier.py rewrite.py decompose.py filters.py router.py
│   ├── graph/         schema.py extract.py loader.py graph_retriever.py
│   ├── fs/            tools.py page_index.py
│   ├── multimodal/    clip_embedder.py caption.py
│   ├── agent/         loop.py tools.py prompts/ answer.py
│   ├── llm/           client.py cache.py cost.py
│   ├── eval/          golden.py synth.py judge.py metrics.py stats.py runner.py deepeval_adapter.py report.py
│   ├── api/           app.py schemas.py
│   ├── ui/            streamlit_app.py
│   ├── testing/       fakes.py fixtures.py
│   └── cli.py         ← Typer: `docseek data|ingest|index|search|ask|eval|bench|serve`
├── tests/
│   ├── unit/  integration/  e2e/   fixtures/mini_corpus/  (khoảng 10 file md nhỏ, commit vào repo)
├── data/              ← gitignored, trừ: internal_kb/, golden/, scans/ground_truth/
├── experiments/       ← kết quả run: <timestamp>_<config>/{metrics.json, per_query.jsonl, traces.jsonl}
├── notebooks/         ← EDA và phân tích failure (không bắt buộc, không chứa logic chính)
└── docs/
    ├── architecture.md
    ├── adr/           0001-own-bm25.md 0002-section-level-relevance.md ...
    ├── milestones/    M0.md ... M8.md      (đề bài cho người học)
    ├── results/       M1.md ... M8.md      (số đo thật + phân tích)
    ├── learning-log/  (của người học)
    └── benchmark_report.md
```

### 4.5. Tech stack

| Mục đích | Chọn | Ghi chú |
|---|---|---|
| Env / deps | uv | `uv sync`, `uv run` |
| CLI | Typer + Rich | |
| Config | pydantic-settings + YAML | `.env` cho key |
| Lexical | **tự cài** + `bm25s` (đối chiếu) + `snowballstemmer` | |
| Embedding | sentence-transformers / fastembed | mặc định `BAAI/bge-small-en-v1.5`, so sánh thêm 1–2 model |
| Sparse learned | fastembed SPLADE (tuỳ chọn ở M4) | |
| Reranker | `BAAI/bge-reranker-base` hoặc `cross-encoder/ms-marco-MiniLM-L-6-v2` | so sánh cả hai |
| Vector DB | Qdrant + `qdrant-client` | Query API (prefetch + fusion) |
| Graph | Neo4j 5 Community + driver `neo4j` | |
| PDF / OCR | PyMuPDF; Tesseract (`pytesseract`) làm baseline; RapidOCR hoặc Docling để so sánh; vision LLM OCR (tuỳ chọn) | đo CER |
| Multimodal | CLIP (`clip-ViT-B-32` qua sentence-transformers) | |
| LLM | LiteLLM | Gemini + OpenAI |
| Eval | tự cài + `ranx` (đối chiếu) + DeepEval | |
| API / UI | FastAPI + Uvicorn / Streamlit | |
| Quality | ruff, mypy (strict cho `core/`, `eval/`), pytest, pytest-cov, pre-commit | |
| Đo RAM / latency | `psutil`, `time.perf_counter`, `tracemalloc` | |

> Claude Code: **kiểm tra version hiện tại và API docs** của DeepEval, qdrant-client, LiteLLM trước khi viết code, vì các thư viện này đổi API khá nhanh. Không hard-code tên model LLM trong code; đặt tên model trong `configs/models.yaml`.

### 4.6. Config và secrets

`.env.example`:
```
GEMINI_API_KEY=
OPENAI_API_KEY=
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me
DOCSEEK_LLM_CACHE=.cache/llm.sqlite
DOCSEEK_BUDGET_USD=5.0        # hard cap; vượt thì dừng và báo lỗi
```

`configs/models.yaml` (ví dụ, tên model đặt theo thời điểm triển khai):
```yaml
llm:
  generator: { model: "gemini/<flash-model>", temperature: 0.7 }
  judge:     { model: "openai/<mini-model>",  temperature: 0.0 }
  agent:     { model: "gemini/<flash-model>", temperature: 0.0 }
  vision:    { model: "gemini/<flash-model>" }
embedding:
  default: "BAAI/bge-small-en-v1.5"
  candidates: ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "intfloat/e5-small-v2"]
reranker:
  default: "BAAI/bge-reranker-base"
```

**Bắt buộc:** không bao giờ commit `.env`. `BudgetGuard` theo dõi tổng `cost_usd` trong mỗi phiên chạy và dừng khi vượt `DOCSEEK_BUDGET_USD`.

---

## 5. Phương pháp đánh giá

Đây là phần "xương sống". Mọi quyết định tối ưu đều phải dựa trên số đo từ đây.

### 5.1. Golden set

- **File:** `data/golden/golden_v{N}.jsonl`, commit vào repo.
  ```json
  {"qid":"q0042","text":"how do I make a dependency that yields and cleans up?",
   "qtype":"howto","principal":"anonymous","split":"test",
   "relevant":{"fastapi-docs:tutorial/dependencies/dependencies-with-yield.md#dependencies-with-yield":2,
               "fastapi-docs:tutorial/dependencies/dependencies-with-yield.md#a-database-dependency-with-yield":1},
   "reference_answer":"...", "origin":"synthetic|human", "reviewed":true}
  ```
- **Thang điểm relevance:** `2` = trả lời trực tiếp; `1` = liên quan hoặc trả lời một phần; không có trong danh sách = `0`.
- **Kích thước mục tiêu:** khoảng **250 query** sau lọc, chia `dev` 40% (dùng để tuning) và `test` 60% (chỉ dùng để báo cáo). **Không tuning trên `test`.**
- **Phân bố theo `qtype`** (để cắt lát kết quả):

| qtype | Ví dụ | Số lượng (xấp xỉ) |
|---|---|---|
| factoid | "default status code for POST response_model?" | 35 |
| howto | "upload multiple files with form data" | 45 |
| conceptual | "why use async def vs def in path operations" | 30 |
| symbol | "`BackgroundTasks`", "`Annotated` Query alias" (exact-term, BM25 nên thắng) | 30 |
| multihop | "which security tutorial uses OAuth2PasswordBearer together with a dependency that yields?" | 20 |
| freshness | "current JWT key rotation interval at Acme" (đáp án nằm ở doc mới nhất) | 15 |
| permission | cùng query nhưng principal khác nhau → tập relevant khác nhau | 15 |
| scan | đáp án chỉ có trong PDF scan (ví dụ bảng SLA) | 20 |
| image | "screenshot showing the Authorize button in Swagger UI" | 15 |
| unanswerable | hỏi thứ không có trong corpus; hệ thống phải nói "không tìm thấy" | 25 |

### 5.2. Sinh synthetic query (`docseek eval synth`)

1. Lấy mẫu section theo tầng: theo `doc_type`, `source` và độ dài.
2. LLM generator sinh query **theo phong cách**:
   - *human*: ngắn, mơ hồ, có lỗi chính tả nhẹ.
   - *machine*: kiểu agent, nhiều keyword.
   - *paraphrase*: tránh lặp lại nguyên văn từ trong section. Đây là cách **giảm bias có lợi cho BM25**.
3. **Bộ lọc:**
   - Bỏ query trùng hoặc gần trùng (cosine > 0.92).
   - Kiểm tra answerability: LLM judge xác nhận section thật sự trả lời được query.
   - Đo **lexical overlap** (tỉ lệ token query xuất hiện trong section) và giữ phân bố cân bằng, ghi phân bố này vào báo cáo.
4. **Hard negatives:** với mỗi query, ghi lại 2–3 section "trông giống nhưng sai" (ví dụ `query-params` và `path-params`). Dùng chúng cho phân tích failure và cho bài kiểm tra reranker.
5. **Human review:** người học tự review **ít nhất 50 query**, sửa hoặc bỏ, và đánh dấu `reviewed: true`. Viết thêm khoảng 30 query thủ công (`origin: human`).

### 5.3. Pooling và golden set v2 (ở M4)

Golden v1 chỉ biết section nguồn là relevant, nên sẽ có **false negative**. Ở M4, khi đã có nhiều retriever:

1. Lấy top-20 của mọi retriever cho mỗi query (pooling kiểu TREC).
2. Dùng LLM judge chấm các cặp (query, section) chưa có nhãn theo thang 0/1/2.
3. Người học review mẫu ngẫu nhiên 100 cặp.
4. Xuất ra `golden_v2`. **Báo cáo metric trên cả v1 và v2** và giải thích sự khác biệt.

### 5.4. LLM-as-judge và hiệu chỉnh (calibration)

- **Rubric** đặt ở `src/docseek/eval/prompts/relevance_judge.md`: tiêu chí rõ cho 0/1/2, có ví dụ few-shot, output JSON `{score, reason}`.
- **Hiệu chỉnh:**
  1. Người học gán nhãn thủ công 100 cặp.
  2. Tính **Cohen's κ** (weighted) giữa judge và người.
  3. Mục tiêu κ ≥ 0.6. Nếu thấp hơn, sửa rubric và lặp lại.
- **Kiểm tra bias:**
  - Position bias: hoán vị thứ tự khi so sánh cặp.
  - Length bias: tương quan giữa độ dài section và điểm.
  - Self-preference: judge khác họ với generator.
- Ghi tất cả vào `docs/results/M1.md`.

### 5.5. Metric (tự cài ở `eval/metrics.py`, kiểm chứng với `ranx`)

Với một query có tập relevant R và danh sách kết quả đã bỏ trùng theo section:

- **Recall@k** = |R ∩ top_k| / |R|
- **Precision@k** = |R ∩ top_k| / k
- **MRR@k** = 1 / rank của kết quả relevant đầu tiên trong top-k (bằng 0 nếu không có)
- **AP@k / MAP@k** = trung bình precision tại mỗi vị trí có kết quả relevant, chia cho min(|R|, k)
- **nDCG@k**: DCG = Σ (2^rel_i − 1) / log2(i + 1); chia cho IDCG
- **Hit@k** (dùng cho query chỉ có 1 relevant)
- **Leakage@k**: số kết quả mà principal **không có quyền** xem. **Bắt buộc bằng 0.**
- **StaleHit@k**: tỉ lệ query `freshness` mà doc `deprecated` hoặc bị `supersedes` xếp **trên** doc mới nhất
- **Abstention accuracy**: với query `unanswerable`, tỉ lệ agent từ chối đúng

**Yêu cầu:**

- Unit test có ví dụ tính tay cho từng metric.
- Có test property-based (hypothesis) cho các bất biến, ví dụ 0 ≤ nDCG ≤ 1, và hoán vị kết quả không relevant thì không làm đổi Recall.
- Một test tích hợp so khớp kết quả với `ranx`, sai số dưới 1e-6.

### 5.6. Thống kê

- **Bootstrap 95% CI** (1.000 lần lấy mẫu theo query) cho mỗi metric.
- So sánh 2 cấu hình bằng **paired randomization test** hoặc paired bootstrap, báo p-value.
- Một cải tiến chỉ được gọi là "tốt hơn" khi Δ > 0 **và** p < 0.05 trên split `test`. Nếu không, ghi là "không khác biệt có ý nghĩa".

### 5.7. Phía answer (M8, dùng DeepEval)

- DeepEval với **judge model tuỳ biến** (wrap LiteLLM thành `DeepEvalBaseLLM`). Các metric: `Faithfulness`, `AnswerRelevancy`, `ContextualPrecision`, `ContextualRecall`, `ContextualRelevancy`, và một `GEval` tuỳ biến cho "citation correctness" (citation có thật sự chứa thông tin được nêu không).
- Metric riêng của agent: số bước trung bình, số tool call, tỉ lệ dùng đúng tool theo `qtype`, latency, cost/query.

### 5.8. Experiment runner và regression gate

- `docseek eval run --config configs/experiments/hybrid_rrf.yaml --split dev` ghi ra `experiments/<ts>_<name>/`:
  - `metrics.json`: tổng thể và theo `qtype`, kèm CI.
  - `per_query.jsonl`: rank của từng relevant, hits.
  - `traces.jsonl`.
  - `env.json`: git sha, config hash, model, version thư viện, thông tin máy.
- `docseek eval compare A B`: bảng Δ, p-value, và danh sách query thắng/thua nhiều nhất (dùng cho phân tích failure).
- `docseek eval gate --baseline experiments/baseline.json`: fail nếu nDCG@10 trên dev giảm hơn 0.01 hoặc Leakage > 0. Chạy trong CI ở chế độ fake (mini corpus + mini golden set) để chặn regression logic.

---

## 6. Milestones

Mỗi milestone gồm: **Mục tiêu → Việc cần làm → Deliverables → Acceptance criteria → Câu hỏi phân tích**. Khi một milestone xong, file `docs/milestones/Mx.md` (đề bài) và `docs/results/Mx.md` (kết quả) phải tồn tại.

> **Về số mục tiêu:** các ngưỡng metric dưới đây là **giả thuyết định hướng**, không phải cam kết. Nếu không đạt, kết quả vẫn được chấp nhận khi có **phân tích nguyên nhân bằng số liệu**. Tuyệt đối không bịa số.

---

### M0 — Foundation & Project Setup *(buổi 01)* · tag `m0-setup`

**Mục tiêu:** khung repo sạch, chạy được, CI xanh, có tài liệu tổng quan về bài toán search.

**Việc cần làm:**

- Tạo `uv init --package`, cấu hình `pyproject.toml`: deps cơ bản, nhóm `dev`, cấu hình ruff/mypy/pytest.
- `docker-compose.yml` gồm Qdrant và Neo4j, có healthcheck và volume.
- `core/config.py`, `core/logging.py` (structlog hoặc logging JSON), `core/models.py`, `core/interfaces.py`, `testing/fakes.py`.
- CLI Typer với lệnh `docseek --help` và `docseek doctor`. `doctor` kiểm tra: Qdrant và Neo4j kết nối được không, key có không (chỉ báo có/không, không in giá trị), Tesseract đã cài chưa.
- `docseek data fetch` (Corpus A) kèm MANIFEST.
- `tests/fixtures/mini_corpus/`: khoảng 10 file md nhỏ có đủ các đặc thù (anchor, code include, admonition, link).
- CI `.github/workflows/ci.yml` gồm: `uv sync` → ruff → mypy → pytest (unit) → coverage report.
- `pre-commit` với ruff và một bước chặn commit file `.env`.
- `docs/architecture.md` (bản nháp), `docs/adr/0001-*.md`, `CLAUDE.md`, `README.md` (khung).
- `docs/overview.md`: bức tranh từ traditional search đến RAG và agentic, các bài toán search hiện đại, và vì sao chọn corpus này.

**Acceptance criteria:**

- [ ] `uv sync && uv run docseek doctor` chạy được trên máy sạch, chỉ cần docker.
- [ ] `docker compose up -d` cho ra Qdrant và Neo4j ở trạng thái healthy.
- [ ] `uv run pytest` xanh, CI xanh trên GitHub.
- [ ] `data/raw/MANIFEST.json` có đúng commit đã pin.

---

### M1 — Ingestion v1 + Evaluation Harness *(buổi 02)* · tag `m1-eval`

**Mục tiêu:** có dữ liệu sạch và **có "thước đo" trước khi tối ưu bất cứ thứ gì**.

**Việc cần làm:**

- `ingest/markdown_parser.py`: tách section theo heading và anchor, resolve code include, chuẩn hoá admonition/HTML, trích xuất link và metadata (§3.1). Mỗi section có `provenance` gồm file và dòng.
- `ingest/chunking.py`, strategy `section` (mặc định): mỗi section là một chunk; section dài hơn 512 token thì tách theo đoạn và có overlap.
- `docseek data gen-internal` (Corpus B) và commit kết quả.
- `docseek ingest` xuất ra `data/processed/sections.jsonl` và `chunks.jsonl`.
- Toàn bộ §5.1, §5.2, §5.4, §5.5, §5.6: synth, golden v1, judge kèm calibration, metrics, stats, runner, compare.
- Retriever giả `RandomRetriever` để kiểm tra đường ống eval (kỳ vọng metric gần 0) và `OracleRetriever` (kỳ vọng Recall@k = 1). Đây là **sanity check** cho chính eval harness.

**Deliverables:** `golden_v1.jsonl` (đã review ≥ 50 query), `docs/results/M1.md` (thống kê corpus, phân bố query, κ của judge, biểu đồ lexical overlap).

**Acceptance criteria:**

- [ ] Parser có unit test cho từng đặc thù. Không còn chuỗi `{* ... *}` nào trong output (có test kiểm tra).
- [ ] Metric khớp `ranx`. Oracle cho Recall = 1, Random cho kết quả gần 0.
- [ ] κ(judge, human) được báo cáo. Nếu dưới 0.6, có ghi lại các vòng sửa rubric.
- [ ] Tổng chi phí LLM cho M1 được ghi vào results (từ trace).

**Câu hỏi phân tích:** Golden set này đang thiên vị retriever nào? Vì sao chấm relevance ở mức section mà không ở mức chunk?

---

### M2 — Lexical Search: TF-IDF → BM25 *(buổi 03)* · tag `m2-bm25`

**Mục tiêu:** tự cài keyword search, hiểu tokenization, scoring và tuning.

**Việc cần làm:**

- `lexical/tokenizer.py` với các tuỳ chọn cấu hình: lowercase, stopword (bật/tắt), stemming (bật/tắt), và **code-aware**:
  - tách `snake_case` và `camelCase`, nhưng **giữ thêm token gốc**, ví dụ `BackgroundTasks` → `backgroundtasks`, `background`, `tasks`;
  - giữ các token như `@app.get` và `response_model`.
- `lexical/inverted_index.py`: postings, df, độ dài doc. Serialize ra đĩa (pickle hoặc msgpack + numpy).
- `lexical/tfidf.py` (baseline) và `lexical/bm25.py` với tham số k1 và b. Hỗ trợ **field weighting** kiểu BM25F đơn giản cho các trường `title`, `section_path`, `body`, `code`.
- Retriever `bm25_own` và `bm25s_lib` (đối chiếu kết quả và tốc độ).
- Grid search k1 ∈ {0.6, 0.9, 1.2, 1.5, 2.0} × b ∈ {0, 0.25, 0.5, 0.75, 1.0} trên **dev**. Vẽ heatmap nDCG@10.
- Ablation: tokenization (có/không stemming, stopword, code-aware), field weights.
- Phân tích failure: 20 query tệ nhất, phân loại nguyên nhân (vocabulary mismatch, từ đồng nghĩa, query quá ngắn, nhiễu từ code block, ...).

**Acceptance criteria:**

- [ ] Điểm BM25 tự cài khớp với một ví dụ tính tay trong unit test.
- [ ] Thứ tự top-10 của `bm25_own` và `bm25s` có overlap ≥ 0.8 khi cùng tokenizer (nếu thấp hơn thì giải thích).
- [ ] Bảng kết quả TF-IDF / BM25 default / BM25 tuned / các ablation, có CI, cắt theo `qtype`.
- [ ] ADR `0001-own-bm25.md`: vì sao tự cài và khi nào nên dùng thư viện.

**Giả thuyết:** BM25 mạnh ở `symbol` và `factoid`, yếu ở `conceptual` và paraphrase.

---

### M3 — Semantic Search with Qdrant *(buổi 04)* · tag `m3-dense`

**Mục tiêu:** dense retrieval, hiểu embedding space, distance metric và payload filter.

**Việc cần làm:**

- `dense/embedders.py`: wrapper sentence-transformers và fastembed. Batch, normalize, cache embedding ra `.npy` theo hash của text và tên model.
- `dense/qdrant_store.py`:
  - Collection với **named vectors** (`dense`; về sau thêm `sparse`, `image`).
  - Payload index cho `acl`, `doc_type`, `source`, `status`, `updated_at`, `modality`.
  - Upsert idempotent theo `chunk_id` (dùng UUID5 làm point id).
- Retriever `dense_<model>` hỗ trợ filter (payload filter của Qdrant). **ACL filter luôn được áp dụng ở mức retriever**, dựa trên principal.
- So sánh 2–3 embedding model: chất lượng, dim, thời gian embed toàn corpus, kích thước index.
- Thử cosine và dot (đã normalize) để thấy khi nào chúng tương đương. Thử có và không có prefix query/passage (với e5/bge).
- So sánh với BM25 theo từng `qtype`. Liệt kê các query mà dense thắng hoặc thua BM25 kèm ví dụ cụ thể.
- Áp dụng ACL filter cho cả BM25: lọc trước khi tính score, hoặc lọc sau khi lấy kết quả nhưng phải over-fetch. Ghi lại tradeoff giữa hai cách.

**Acceptance criteria:**

- [ ] Integration test với `QdrantClient(":memory:")` và `HashEmbedder`.
- [ ] Leakage@10 = 0 cho mọi principal (có test riêng với các query `permission`).
- [ ] Bảng BM25 và dense theo `qtype` kèm p-value.

---

### M4 — Hybrid Retrieval + Reranking *(buổi 05)* · tag `m4-hybrid`

**Mục tiêu:** kết hợp nhiều tín hiệu retrieval, rerank, và đo **quality lift so với latency**.

**Việc cần làm:**

- `hybrid/normalization.py`: min-max, z-score.
- `hybrid/fusion.py`:
  - **RRF tự cài** (tham số k = 60, có tuning).
  - Weighted sum sau normalize (tham số α, có tuning).
  - Gọi Qdrant Query API dùng `prefetch` với fusion RRF/DBSF để so sánh.
- (Tuỳ chọn) sparse learned vector (SPLADE) đặt trong cùng collection.
- `rerank/cross_encoder.py`: rerank top-N với N ∈ {20, 50, 100}. Đo latency p50/p95 trên CPU và vẽ đường **nDCG@10 theo latency**.
- (Tuỳ chọn) `rerank/llm_reranker.py` (listwise, qua API). So sánh chất lượng, latency và cost với cross-encoder.
- **Golden set v2** bằng pooling (§5.3). Báo cáo trên cả v1 và v2.
- Kiểm tra reranker có tách được hard negatives không: tỉ lệ hard negative bị đẩy xuống dưới positive.

**Acceptance criteria:**

- [ ] Có unit test cho RRF (ví dụ tính tay) và cho các trường hợp biên: list rỗng, trùng id, tie.
- [ ] Hybrid tốt nhất ≥ max(BM25, dense) về nDCG@10 trên test. Nếu không, phải giải thích bằng số liệu.
- [ ] Biểu đồ quality–latency và khuyến nghị cấu hình mặc định, có lý do.
- [ ] `golden_v2.jsonl` kèm ghi chú review.

---

### M5 — Query Understanding, Metadata & Chunking *(buổi 06)* · tag `m5-query`

**Mục tiêu:** search harness "hiểu" query: rewrite, decomposition, metadata filter, router, và tối ưu chunking.

**Việc cần làm:**

- `query/classifier.py`: phân loại `qtype`. Bước 1 là luật rẻ (backtick, CamelCase → `symbol`; "latest/current" → `freshness`). Bước 2 là LLM với structured output. Đo **accuracy của classifier** trên golden set (vì đã có nhãn `qtype`).
- `query/filters.py`: LLM trích xuất filter từ query (`doc_type`, `source`, khoảng thời gian, version) thành `MetadataFilter`. Validate bằng pydantic. Nếu filter làm rỗng kết quả thì **fallback** bỏ filter, và ghi lại việc fallback vào trace.
- `query/rewrite.py`: multi-query rewrite, (tuỳ chọn) HyDE, và mở rộng từ đồng nghĩa cho BM25.
- `query/decompose.py`: tách query `multihop` thành các sub-query, chạy song song, rồi fuse.
- `query/router.py`: map `qtype` sang pipeline, ví dụ:
  - `symbol` → BM25 có trọng số cao;
  - `conceptual` → dense + rerank;
  - `freshness` → hybrid + **freshness boost**;
  - `multihop` → decompose (M6 bổ sung graph).
- **Metadata là "retriever thứ ba":**
  - Freshness boost `score × (1 + β·exp(−Δdays/τ))`.
  - Phạt điểm khi `status=deprecated` hoặc doc bị `supersedes`.
  - Authority boost theo `doc_type`.
  - Đo **StaleHit@k** trước và sau.
- **Ablation chunking:**
  - `fixed_256` (cắt theo token);
  - `section` (M1);
  - `section+context_header` (chèn `section_path` và title vào `text_for_embedding`);
  - `semantic` (tuỳ chọn).
  - So sánh trên **cùng golden set ở mức section**. Đây là lúc quyết định ở §3.5 phát huy tác dụng.

**Acceptance criteria:**

- [ ] Bảng lift theo từng thành phần (ablation: tắt từng phần một), cắt theo `qtype`.
- [ ] StaleHit@5 giảm có ý nghĩa thống kê khi bật freshness và supersedes.
- [ ] Có test fallback khi filter rỗng. Leakage vẫn bằng 0 với mọi đường đi của router.
- [ ] ADR về chunking strategy được chọn.

---

### M6 — Knowledge Graph + Filesystem Search *(buổi 07)* · tag `m6-graph-fs`

**Mục tiêu:** retrieval theo quan hệ (multi-hop) và cho agent "điều hướng" corpus như một developer.

**Graph (Neo4j):**

- **Schema** (ghi trong `graph/schema.py` và `docs/architecture.md`):
  ```
  (:Page {doc_id, title, doc_type})   (:Section {section_id, title, path})
  (:APIElement {name, kind})          // Depends, HTTPException, BackgroundTasks, OAuth2PasswordBearer...
  (:Concept {name})                   // dependency injection, OAuth2, CORS, ...
  (:CodeExample {path})               // docs_src/*.py
  (:Version {name})                   // từ release-notes
  (:KBDoc {id, acl, status, updated_at})

  (Section)-[:PART_OF]->(Page)        (Page)-[:LINKS_TO]->(Page)
  (Section)-[:MENTIONS]->(APIElement|Concept)
  (Section)-[:SHOWS]->(CodeExample)   (CodeExample)-[:USES]->(APIElement)
  (APIElement)-[:CHANGED_IN]->(Version)
  (KBDoc)-[:SUPERSEDES]->(KBDoc)      (KBDoc)-[:REFERENCES]->(APIElement|Concept)
  ```
- **Extraction** (`graph/extract.py`), kết hợp nhiều nguồn:
  - Rule-based: link markdown, identifier trong backtick.
  - **AST của `docs_src/*.py`**: các `import` và tên được gọi, rồi map sang `APIElement`. Cách này chính xác và miễn phí, không cần LLM.
  - LLM, có cache, để trích `Concept` và quan hệ `RELATED_TO`.
  - Entity resolution: chuẩn hoá tên và gộp alias.
- `graph/graph_retriever.py`:
  1. Entity linking từ query (exact/fuzzy match tên node, có thể dùng thêm embedding của tên node).
  2. **Cypher template có tham số** cho các mẫu multi-hop. Không để LLM tự sinh Cypher tuỳ ý (rủi ro injection và khó đánh giá). Ghi điều này vào ADR.
  3. Trả về các section, kèm đường đi trong graph làm "giải thích".
- ACL vẫn được áp dụng: lọc `KBDoc.acl` ngay trong Cypher.

**Filesystem search:**

- `ingest/projection.py`: chiếu corpus thành cây markdown `data/projected/`, mỗi section là một heading có dòng `<!-- section_id: ... -->`. Thêm `_index.md` (page index) cho mỗi thư mục, liệt kê trang, `section_path` và số dòng.
- `fs/tools.py`: các tool `ls(path)`, `find(glob)`, `grep(pattern, path, regex=False, max_results=50)`, `cat(path, start_line, end_line | section_id)`.
  - **Sandbox:** chặn path traversal (`..`, symlink).
  - **Áp ACL theo principal:** file mà principal không được xem thì coi như không tồn tại.
  - Giới hạn kích thước output.
- Mỗi tool call được ghi trace.

**Acceptance criteria:**

- [ ] Có test path traversal, test ACL cho FS tools, và test các Cypher template trên graph nhỏ dựng bằng fixture.
- [ ] Trên tập `multihop`: graph (hoặc graph + hybrid) cải thiện Recall@10 so với hybrid M5, hoặc có phân tích vì sao không.
- [ ] Phần "When graph beats vector retrieval (and when it doesn't)" trong results, có ví dụ cụ thể.

---

### M7 — Document Parsing (OCR) + Multimodal *(buổi 08)* · tag `m7-multimodal`

**Mục tiêu:** đưa PDF scan, bảng và ảnh vào cùng một retrieval experience. Đo chất lượng parsing bằng số.

**Việc cần làm:**

- `docseek data make-scans` (Corpus C, §3.3).
- `ingest/pdf_parser.py`: phát hiện trang có text layer hay không. Nếu có thì dùng PyMuPDF; nếu không thì OCR.
- `ingest/ocr.py`: so sánh **Tesseract** với **RapidOCR** hoặc **Docling**, và (tuỳ chọn) **vision LLM OCR**.
  - Đo **CER/WER** so với ground truth, thời gian mỗi trang, và cost mỗi trang.
  - Xử lý reading order cho trang 2 cột.
  - Trích bảng ra markdown table. Đo độ chính xác cell trên các bảng SLA và Error codes.
- Chunk từ scan mang provenance `page` và (nếu có) `bbox`. Câu trả lời trích dẫn được "trang X của file Y".
- `multimodal/`:
  - Embedding ảnh bằng CLIP vào named vector `image`.
  - Caption bằng vision LLM (có cache).
  - OCR text của screenshot.
  - **Ba cách retrieval ảnh:**
    1. text → CLIP image vector (cross-modal);
    2. text → caption + OCR (đưa ảnh về text);
    3. fusion của cả hai.
  - Đo trên tập query `image`.
- Đưa các retriever mới vào router: `qtype=scan` dùng hybrid trên chunk scan; `qtype=image` dùng image retriever.

**Acceptance criteria:**

- [ ] Bảng OCR: engine × CER × WER × giây/trang × $/trang × độ chính xác bảng.
- [ ] Bảng image retrieval với 3 phương án, kèm nhận xét chi phí (caption tốn tiền; CLIP miễn phí nhưng yếu với text trong ảnh).
- [ ] Query `scan` và `image` có trong eval tổng, Leakage vẫn bằng 0.

---

### M8 — Agent, API, UI, Benchmark & Ship *(Final project)* · tag `v1.0.0`

**Mục tiêu:** ghép tất cả thành sản phẩm demo được, có báo cáo production tradeoff và README chuẩn CV.

**Agent (`agent/`):**

- **Tự cài agent loop** với tool calling qua LiteLLM:
  - `max_steps` (mặc định 6), timeout, và budget theo từng query.
  - Tools: `search(query, mode=auto|bm25|dense|hybrid, filters)`, `graph_search(entity, pattern)`, `ls`, `find`, `grep`, `cat`, `get_section(section_id)`, `image_search(query)`.
  - Tool description viết kỹ, vì đây chính là "search skills" của agent.
  - Principal được **inject từ server**. Agent không tự đặt hay đổi principal được (có test chứng minh).
- **Trả lời:**
  - Bắt buộc có citation `[section_id]`. Có bước hậu kiểm: citation phải nằm trong các section đã retrieve, nếu không thì loại.
  - Nếu không đủ bằng chứng thì **abstain** ("I couldn't find this in the docs"), kèm gợi ý các section gần nhất.
- So sánh 3 chế độ: (a) one-shot RAG: router + hybrid + rerank rồi gọi LLM; (b) agentic; (c) agentic không có FS tools. Đo answer quality (DeepEval), số bước, latency và cost.

**API (FastAPI):**

- `POST /search` và `POST /ask` nhận `{query, principal, mode, k, debug}`. `principal` lấy từ header `X-Principal` (auth giả lập, ghi rõ là demo).
- `GET /health`, `GET /traces/{trace_id}`.
- Pydantic schema, OpenAPI docs, test bằng `TestClient`.

**UI (Streamlit):**

- Ô tìm kiếm, chọn principal, chọn mode.
- Tab **Compare**: kết quả BM25 / dense / hybrid / rerank đặt cạnh nhau.
- Tab **Answer**: câu trả lời có citation bấm được, hiện đoạn nguồn và trang PDF.
- Tab **Trace**: timeline các bước, tool call, latency và cost.
- Tab **Eval**: đọc `experiments/` và hiển thị bảng/biểu đồ.

**Benchmark (`docseek bench`) → `docs/benchmark_report.md`:**

- Với mỗi cấu hình (BM25, dense, hybrid, hybrid+rerank, full router, agentic), ghi:
  - Quality: Recall@10, MRR@10, nDCG@10 (kèm CI), cắt theo `qtype`.
  - Latency: p50/p95/p99, chạy warm, 3 lần lặp.
  - Tài nguyên: RAM của process và container Qdrant/Neo4j (`docker stats`), kích thước index trên đĩa, thời gian build index.
  - Cost: $ cho 1.000 query (LLM), $ ingest một lần (caption, internal KB, judge).
- **Tradeoff thêm:** Qdrant scalar quantization int8 (RAM so với chất lượng), HNSW `ef` (latency so với recall).
- **Failure analysis:** phân loại các lỗi còn lại (dữ liệu / retrieval / ranking / generation), mỗi loại có ví dụ. Kèm "what I'd do next".
- Ghi rõ cấu hình máy chạy benchmark.

**Đóng gói:**

- `docker compose up` chạy được qdrant, neo4j, api, ui.
- `make demo` chạy fetch → ingest → index → mở UI. Có sẵn đường "quick demo" dùng index dựng sẵn từ mini corpus, chạy dưới 2 phút.
- README hoàn chỉnh (§9), sơ đồ kiến trúc (Mermaid hoặc ảnh), GIF hoặc ảnh chụp demo.
- Release `v1.0.0` trên GitHub, kèm changelog.

**Acceptance criteria:**

- [ ] Clone sạch → `docker compose up` → demo chạy được theo README.
- [ ] Benchmark report đủ các cột trên với **số đo thật**.
- [ ] Coverage ≥ 80% cho `core/`, `eval/`, `lexical/`, `hybrid/`, `fs/`; tổng thể ≥ 70%.
- [ ] Agent: có test chứng minh không leak ACL, không vượt `max_steps`, và citation hợp lệ.

---

## 7. Tiêu chuẩn kỹ thuật (áp dụng mọi milestone)

- **Typing:** type hint đầy đủ. `mypy --strict` cho `core/`, `eval/`, `lexical/`, `hybrid/`.
- **Style:** ruff (lint + format), line length 100. Docstring kiểu Google cho các API công khai.
- **Tests:**
  - `tests/unit` chạy nhanh, không I/O mạng.
  - `tests/integration` có marker `@pytest.mark.integration`, cần docker.
  - `tests/e2e` có marker `e2e`, cần key, không chạy trong CI.
  - Test không bao giờ gọi LLM thật. Dùng `ScriptedLLM` hoặc cache fixture.
- **Determinism:** seed cố định cho mọi bước có random (lấy mẫu, bootstrap, synth). Ghi seed vào `env.json`.
- **Logging:** structured logging. Không log nội dung key. Mọi query có `trace_id`.
- **Hiệu năng:** embedding chạy theo batch, có cache. Index build idempotent; chạy lại thì bỏ qua phần không đổi, dựa trên hash.
- **Bảo mật:** sandbox cho FS tools. Cypher chỉ dùng template có tham số. ACL được enforce ở **tầng retriever**, không dựa vào prompt. Tool output đưa vào LLM được bọc delimiter, và có ghi chú về rủi ro prompt injection từ nội dung tài liệu.
- **Lỗi:** khi API LLM lỗi thì retry với exponential backoff (LiteLLM). Nếu vẫn lỗi thì degrade: router rơi về hybrid, agent rơi về one-shot RAG.
- **Không được:** commit secret; commit `data/raw` hoặc index lớn; dùng `print` thay logging trong thư viện; để TODO không có issue tương ứng.

---

## 8. CI/CD

`.github/workflows/ci.yml`:

1. `lint`: `ruff check`, `ruff format --check`, `mypy`.
2. `unit`: pytest unit + coverage, upload artifact coverage.
3. `integration`: service container `qdrant/qdrant` và `neo4j:5-community`; pytest `-m integration` trên mini corpus với `HashEmbedder`.
4. `eval-gate`: chạy `docseek eval run` trên mini corpus và mini golden set (fake), so với `tests/fixtures/baseline_metrics.json`. Fail khi metric giảm hoặc Leakage > 0.

Badge CI và coverage đặt trên README.

---

## 9. README và tài liệu cho CV

### 9.1. Cấu trúc README (tiếng Anh)

1. Tên, một câu mô tả, badges.
2. **Results at a glance:** bảng nhỏ BM25 → dense → hybrid → +rerank → +router → agentic với nDCG@10, Recall@10, p95 latency và $/1k queries (**số thật**).
3. Demo GIF.
4. Architecture: sơ đồ, kèm 5–7 gạch đầu dòng về các thành phần.
5. What makes it production-grade: evaluation methodology, ACL enforcement, freshness, tracing, cost tracking, CI regression gate.
6. Quickstart (≤ 5 lệnh) và Full pipeline.
7. Project structure.
8. Key engineering decisions (link tới ADR).
9. Benchmark report (link) và failure analysis.
10. Roadmap / what I'd do next.
11. Acknowledgements: FastAPI docs (MIT); syllabus tham khảo từ khoá học "Production-Grade Agentic RAG".

### 9.2. Mẫu bullet cho CV (chỉ điền số đo thật)

- Built an agentic retrieval system over ~X sections of technical docs (text, scanned PDFs, images). Improved nDCG@10 from **A** (BM25) to **B** (hybrid + cross-encoder + query routing), p < 0.05, on a 250-query golden set.
- Designed an evaluation harness: synthetic query generation, LLM-as-judge calibrated against human labels (Cohen's κ = **K**), bootstrap CIs, and a CI regression gate.
- Implemented BM25 from scratch with code-aware tokenization; tuned k1/b and field weights (+**Δ** nDCG@10 on symbol queries).
- Enforced document-level ACL across vector, lexical, graph and filesystem retrieval (0 leakage across N permission tests); added freshness-aware ranking that cut stale hits by **S%**.
- Benchmarked OCR engines on synthetic scans (CER **c1** vs **c2**) and multimodal retrieval strategies; reported latency/RAM/cost tradeoffs incl. int8 quantization.

### 9.3. Chuẩn bị phỏng vấn

Tạo `docs/interview-notes.md` gồm 15–20 câu hỏi kèm câu trả lời ngắn, dựa trên chính dự án. Ví dụ:
- Vì sao chọn RRF thay vì weighted sum?
- Làm sao biết LLM judge đáng tin?
- Vì sao chấm relevance ở mức section?
- Khi nào graph thắng vector?
- Lỗi lớn nhất còn lại của hệ thống là gì?

---

## 10. Rủi ro và cách giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Chi phí LLM vượt kiểm soát | Cache mọi call, `BudgetGuard`, model rẻ cho generator, chỉ chạy judge trên phần chưa có nhãn |
| Golden set thiên vị (lexical overlap, false negative) | Sinh query paraphrase, đo overlap, pooling v2, human review |
| Overfit vào benchmark | Tách dev/test, chỉ báo cáo trên test, kiểm định thống kê |
| Thư viện đổi API (DeepEval, qdrant-client, LiteLLM) | Pin version trong `uv.lock`, adapter layer, kiểm tra docs khi nâng cấp |
| FastAPI docs thay đổi cấu trúc | Pin commit; parser có test trên fixture |
| Máy yếu (CPU, RAM) | Mặc định dùng model small; giới hạn số ảnh; rerank window có thể chỉnh |
| Windows | Khuyến nghị WSL2; CLI viết bằng Python (Typer) nên chạy được mọi nơi, Makefile chỉ là alias |
| Scope quá lớn | Mỗi milestone tự đứng được, có tag. Các phần tuỳ chọn (SPLADE, LLM reranker, vision OCR, semantic chunking) chỉ làm khi phần chính đã xong |

---

## 11. Ước lượng công sức cho người học (tự code)

| M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | Tổng |
|---|---|---|---|---|---|---|---|---|---|
| 0.5–1d | 3–4d | 2d | 2d | 2–3d | 3d | 3–4d | 3d | 4–5d | ~5–6 tuần part-time |

---

## Phụ lục A — Prompt mẫu cho từng phiên Claude Code

**Mở đầu mọi phiên:**
```
Đọc PROJECT_SPEC.md và CLAUDE.md. Chúng ta đang làm milestone M{x}.
1) Tóm tắt lại phạm vi M{x} và acceptance criteria. Liệt kê những gì đã có từ milestone trước (đọc code thật, đừng giả định).
2) Đề xuất plan chia thành các bước nhỏ, mỗi bước kết thúc bằng test xanh và một commit.
3) Kiểm tra version và API hiện tại của các thư viện sẽ dùng trước khi viết code.
Chờ tôi duyệt plan rồi mới code.
```

**Kết thúc milestone:**
```
Chạy toàn bộ test, lint, mypy. Chạy eval cho M{x} (split dev để tuning, split test để báo cáo).
Viết docs/milestones/M{x}.md (đề bài cho người tự code, KHÔNG chứa lời giải)
và docs/results/M{x}.md (số đo thật, bảng, biểu đồ, failure analysis, chi phí LLM).
Cập nhật README nếu cần. Commit, tạo tag m{x}-<slug>, push.
```

**Review code người học tự viết:**
```
So sánh nhánh my/m{x} với tag m{x}-<slug>. Đánh giá theo: đúng acceptance criteria,
chất lượng code (typing, test, cấu trúc), hiệu năng, và kết quả eval (dùng eval compare, có p-value).
Chỉ ra 3 điểm tôi làm tốt hơn, 3 điểm nên học từ bản tham chiếu, và các bug tiềm ẩn. Không sửa code của tôi.
```

## Phụ lục B — Nội dung đề xuất cho `CLAUDE.md`

```markdown
# CLAUDE.md
- Source of truth: PROJECT_SPEC.md. Đọc trước mỗi phiên. Làm đúng milestone hiện tại, không làm trước milestone sau.
- Python 3.12 + uv. Lệnh: `uv run pytest`, `uv run ruff check . && uv run ruff format --check .`, `uv run mypy src`.
- Không gọi LLM thật trong test; dùng fakes ở src/docseek/testing/fakes.py.
- Mọi LLM call đi qua CachedLLMClient + BudgetGuard. Không hard-code tên model; dùng configs/models.yaml.
- ACL phải được enforce ở tầng retriever/tool, không bao giờ chỉ dựa vào prompt.
- Relevance chấm ở mức section_id; chunk được map về section khi eval.
- Không bao giờ bịa số liệu trong docs/results hay README; chỉ ghi số từ experiments/.
- Commit nhỏ theo Conventional Commits; mỗi milestone kết thúc bằng tag m{x}-<slug>.
- Không commit .env, data/raw, index; giữ data/golden, data/internal_kb, data/scans/ground_truth.
- Khi không chắc về một quyết định thiết kế có ảnh hưởng lớn: hỏi trước, sau đó ghi ADR.
```

## Phụ lục C — Bảng thuật ngữ nhanh

| Thuật ngữ | Ý nghĩa ngắn |
|---|---|
| BM25 (k1, b) | Scoring keyword: k1 điều khiển mức bão hoà của term frequency, b điều khiển mức chuẩn hoá theo độ dài doc |
| RRF | Reciprocal Rank Fusion: Σ 1/(k + rank). Không cần normalize score |
| DBSF | Distribution-Based Score Fusion: chuẩn hoá score theo phân phối rồi cộng |
| Bi-encoder / Cross-encoder | Encode query và doc riêng (nhanh, dùng để retrieve) / encode cặp query–doc cùng lúc (chậm, chính xác, dùng để rerank) |
| Pooling | Gộp top-k của nhiều hệ thống để gán nhãn, giảm false negative |
| Cohen's κ | Mức đồng thuận giữa hai người (hoặc model) gán nhãn, đã loại trừ phần đồng thuận do ngẫu nhiên |
| CER / WER | Character / Word Error Rate của OCR |
| HyDE | Sinh câu trả lời giả định, embed nó, rồi dùng để search |
| Leakage | Kết quả trả về mà principal không có quyền xem |
