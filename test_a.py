from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "R2AIStage1DATA.json"
DEFAULT_OUT_DIR = BASE_DIR / "submission_eval"
TECHNICAL_BLOCK_RE = re.compile(
    r"\s*<APPLIED_EVIDENCE_JSON>.*?</APPLIED_EVIDENCE_JSON>\s*",
    re.DOTALL,
)
ARTICLE_DIEU_SUFFIX_RE = re.compile(
    r"(\|Điều\s+\d+[A-Za-zĐđ]?)\s*\([^|]*\)\s*$",
    re.IGNORECASE,
)
LEGACY_ERROR_ANSWER_PREFIX = "Không thể tạo câu trả lời do lỗi hệ thống"


def make_nvidia_get_llm(
    model: str,
    base_url: str,
    max_tokens: int,
    api_keys: list[str] | None = None,
    base_urls: list[str] | None = None,
):
    from langchain_openai import ChatOpenAI

    keys = [key.strip() for key in (api_keys or []) if key and key.strip()]
    if not keys:
        keys = [
            key.strip()
            for key in os.getenv("NVIDIA_API_KEYS", "").split(",")
            if key.strip()
        ]
    if not keys:
        single_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if single_key:
            keys = [single_key]
    if not keys:
        raise RuntimeError("Missing NVIDIA_API_KEY or NVIDIA_API_KEYS for --llm-backend nvidia.")

    urls = [url.strip() for url in (base_urls or []) if url and url.strip()]
    if not urls:
        urls = [
            url.strip()
            for url in os.getenv("NVIDIA_LLM_BASE_URLS", "").split(",")
            if url.strip()
        ]
    if not urls:
        urls = [base_url]

    if len(urls) == 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            "Number of NVIDIA base URLs must be 1 or equal to the number of NVIDIA API keys. "
            f"Got {len(urls)} URLs and {len(keys)} keys."
        )

    endpoint_pool = [
        {
            "api_key": key,
            "base_url": urls[idx],
            "name": f"nvidia:{idx + 1}",
        }
        for idx, key in enumerate(keys)
    ]
    pool_lock = threading.Lock()
    next_endpoint = 0

    def pick_endpoint() -> dict[str, str]:
        nonlocal next_endpoint
        with pool_lock:
            endpoint = endpoint_pool[next_endpoint % len(endpoint_pool)]
            next_endpoint += 1
            return endpoint

    def get_nvidia_llm(
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 20,
        enable_thinking: bool = True,
        thinking_token_budget: int | None = None,
    ) -> ChatOpenAI:
        endpoint = pick_endpoint()

        return ChatOpenAI(
            model=model,
            base_url=endpoint["base_url"],
            api_key=endpoint["api_key"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=3,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                }
            },
        )

    get_nvidia_llm.pool_size = len(endpoint_pool)
    get_nvidia_llm.pool_summary = ", ".join(
        f"{endpoint['name']}@{endpoint['base_url']}"
        for endpoint in endpoint_pool
    )
    return get_nvidia_llm


def parse_csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]

def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Input must be a JSON array: {path}")

    questions = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{idx} is not an object")
        if "id" not in item or "question" not in item:
            raise ValueError(f"Item #{idx} must contain id and question")
        questions.append({"id": item["id"], "question": str(item["question"])})
    return questions


def is_completed_result(item: dict[str, Any]) -> bool:
    answer = str(item.get("answer") or "").strip()
    return bool(answer) and not answer.startswith(LEGACY_ERROR_ANSWER_PREFIX)


def load_existing_results(path: Path) -> dict[Any, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Existing results must be a JSON array: {path}")
    return {
        item.get("id"): item
        for item in data
        if isinstance(item, dict) and is_completed_result(item)
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def clean_article_ref(ref: str) -> str:
    return ARTICLE_DIEU_SUFFIX_RE.sub(r"\1", str(ref or "").strip())


def clean_submission_result(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(result)
    cleaned["relevant_docs"] = unique_keep_order(cleaned.get("relevant_docs", []) or [])
    cleaned["relevant_articles"] = unique_keep_order(
        [clean_article_ref(ref) for ref in cleaned.get("relevant_articles", []) or []]
    )
    return cleaned


def final_ai_answer(state: dict[str, Any]) -> str:
    for msg in reversed(state.get("messages", []) or []):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            answer = str(getattr(msg, "content", "") or "").strip()
            return TECHNICAL_BLOCK_RE.sub("", answer).strip()
    return ""


class SubmissionRefResolver:
    def __init__(self, es_host: str, index_name: str):
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        self._source_cache: dict[str, dict[str, Any] | None] = {}

    def _load_sources(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        ids_to_fetch = [cid for cid in chunk_ids if cid not in self._source_cache]
        if ids_to_fetch:
            docs = self.es.mget(index=self.index_name, body={"ids": ids_to_fetch}).get("docs", [])
            for doc in docs:
                doc_id = str(doc.get("_id") or "")
                self._source_cache[doc_id] = doc.get("_source") if doc.get("found") else None

            returned_ids = {str(doc.get("_id") or "") for doc in docs}
            for missing_id in set(ids_to_fetch) - returned_ids:
                self._source_cache[missing_id] = None

        return [
            self._source_cache[cid]
            for cid in chunk_ids
            if self._source_cache.get(cid)
        ]

    def to_submission_refs(self, chunk_ids: list[str]) -> tuple[list[str], list[str]]:
        sources = self._load_sources(unique_keep_order(chunk_ids))
        relevant_docs = []
        relevant_articles = []

        for src in sources:
            document_number = str(src.get("document_number") or "").strip()
            doc_name = str(src.get("raw_title") or "").strip()
            article_no = str(src.get("article_no") or "").strip()

            if document_number and doc_name:
                relevant_docs.append(f"{document_number}|{doc_name}")
            if document_number and doc_name and article_no:
                relevant_articles.append(f"{document_number}|{doc_name}|{article_no}")

        return unique_keep_order(relevant_docs), unique_keep_order(relevant_articles)


def choose_chunk_ids(state: dict[str, Any], prefer_applied: bool = True) -> list[str]:
    applied_ids = state.get("applied_chunk_ids", []) or []
    candidate_ids = state.get("relevant_chunk_ids", []) or []
    if prefer_applied and applied_ids:
        return unique_keep_order(applied_ids)
    return unique_keep_order(candidate_ids)


def make_submission_zip(results_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")


def select_questions(
    questions: list[dict[str, Any]],
    limit: int | None,
    start_id: int | None,
    end_id: int | None,
    ids: set[int] | None,
) -> list[dict[str, Any]]:
    selected = []
    for item in questions:
        qid = int(item["id"])
        if ids is not None and qid not in ids:
            continue
        if start_id is not None and qid < start_id:
            continue
        if end_id is not None and qid > end_id:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def run_one_item(app, item: dict[str, Any], prefer_applied: bool) -> dict[str, Any]:
    started_at = time.time()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"question_id": item["id"], "question": item["question"]}, config)

    chunk_ids = choose_chunk_ids(state, prefer_applied=prefer_applied)
    answer = final_ai_answer(state)
    if not answer:
        raise ValueError("Graph finished without a final answer")

    diagnostics = {
        "id": item["id"],
        "thread_id": thread_id,
        "used_chunk_ids": chunk_ids,
        "applied_chunk_ids": state.get("applied_chunk_ids", []) or [],
        "relevant_chunk_ids": state.get("relevant_chunk_ids", []) or [],
        "candidate_but_not_applied_chunk_ids": state.get("candidate_but_not_applied_chunk_ids", []) or [],
        "evidence_selection_notes": state.get("evidence_selection_notes", ""),
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": answer,
        "chunk_ids": chunk_ids,
        "diagnostics": diagnostics,
    }


def format_question_preview(question: str, max_chars: int = 100) -> str:
    question = " ".join(str(question or "").split())
    if len(question) <= max_chars:
        return question
    return question[:max_chars].rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legal RAG graph on R2AI questions and create results.json/submission.zip."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-name", default="results.json")
    parser.add_argument("--zip-name", default="submission.zip")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--ids", help="Comma-separated question ids, e.g. 1,2,10")
    parser.add_argument("--force", action="store_true", help="Re-run questions already present in results.json")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--use-relevant", action="store_true", help="Use relevant_chunk_ids instead of applied_chunk_ids")
    parser.add_argument("--workers", type=int, default=1, help="Number of questions to run concurrently")
    parser.add_argument("--graph-mode", choices=("stateless", "persistent"), default="stateless")
    parser.add_argument("--stateless", action="store_true", help="Deprecated alias for --graph-mode stateless")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between questions")
    parser.add_argument("--llm-backend", choices=("default", "nvidia"), default=os.getenv("EVAL_LLM_BACKEND", "default"))
    parser.add_argument("--nvidia-model", default=os.getenv("NVIDIA_LLM_MODEL", "qwen/qwen3.5-122b-a10b"))
    parser.add_argument("--nvidia-base-url", default=os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument(
        "--nvidia-base-urls",
        default=os.getenv("NVIDIA_LLM_BASE_URLS"),
        help="Comma-separated NVIDIA base URLs. Use one URL for all keys or one URL per key.",
    )
    parser.add_argument(
        "--nvidia-api-keys",
        default=os.getenv("NVIDIA_API_KEYS"),
        help="Comma-separated NVIDIA API keys. Prefer env var to avoid shell history.",
    )
    parser.add_argument("--nvidia-max-tokens", type=int, default=int(os.getenv("NVIDIA_LLM_MAX_TOKENS", "8192")))
    args = parser.parse_args()
    if not args.input.is_absolute():
        args.input = BASE_DIR / args.input
    if not args.out_dir.is_absolute():
        args.out_dir = BASE_DIR / args.out_dir
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.stateless:
        args.graph_mode = "stateless"

    filter_stats_path = args.out_dir / "filter_stats.jsonl"
    os.environ["FILTER_STATS_LOG_PATH"] = str(filter_stats_path)
    if args.force and filter_stats_path.exists():
        filter_stats_path.unlink()

    questions = load_questions(args.input)
    selected = select_questions(
        questions,
        limit=args.limit,
        start_id=args.start_id,
        end_id=args.end_id,
        ids=parse_ids(args.ids),
    )

    results_path = args.out_dir / args.results_name
    zip_path = args.out_dir / args.zip_name
    diagnostics_path = args.out_dir / "diagnostics.jsonl"
    errors_path = args.out_dir / "errors.jsonl"

    existing_by_id = {} if args.force else load_existing_results(results_path)
    results_by_id = dict(existing_by_id)
    print(f"[Resume] results_path={results_path}")
    print(f"[Logs] filter_stats_path={filter_stats_path}")
    print(f"[Resume] completed_existing={len(existing_by_id)} force={args.force}")

    if args.graph_mode == "stateless":
        os.environ["CHECKPOINTER_BACKEND"] = "none"

    sys.path.insert(0, str(BASE_DIR))
    from Agents.config import (
        CHECKPOINTER_BACKEND,
        COMPRESS_LLM_ENABLE_THINKING,
        EMBEDDING_MAX_CONCURRENT,
        ES_HOST,
        INDEX_NAME,
        LLM_BASE_URL,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        RERANKER_BATCH_SIZE,
        RERANKER_MAX_CONCURRENT,
        VNCORENLP_MAX_CONCURRENT,
    )
    from Agents import graph as graph_module
    from Agents.tools import search_legal as search_legal_module

    if args.llm_backend == "nvidia":
        nvidia_get_llm = make_nvidia_get_llm(
            model=args.nvidia_model,
            base_url=args.nvidia_base_url,
            max_tokens=args.nvidia_max_tokens,
            api_keys=parse_csv_values(args.nvidia_api_keys),
            base_urls=parse_csv_values(args.nvidia_base_urls),
        )
        graph_module.get_llm = nvidia_get_llm
        search_legal_module.get_llm = nvidia_get_llm
        print(
            f"[LLM] Using NVIDIA endpoint pool model={args.nvidia_model} "
            f"pool_size={nvidia_get_llm.pool_size} endpoints={nvidia_get_llm.pool_summary}"
        )
    else:
        print(f"[LLM] Using default endpoint model={LLM_MODEL} base_url={LLM_BASE_URL} max_tokens={LLM_MAX_TOKENS}")

    print(
        "[Runtime] "
        f"workers={args.workers} graph_mode={args.graph_mode} "
        f"checkpointer={CHECKPOINTER_BACKEND} "
        f"compress_thinking={COMPRESS_LLM_ENABLE_THINKING} "
        f"embedding_concurrency={EMBEDDING_MAX_CONCURRENT} "
        f"reranker_concurrency={RERANKER_MAX_CONCURRENT} "
        f"reranker_batch_size={RERANKER_BATCH_SIZE} "
        f"vncorenlp_concurrency={VNCORENLP_MAX_CONCURRENT}"
    )

    app = graph_module.app
    stateless_app = graph_module.stateless_app

    if args.graph_mode == "persistent" and CHECKPOINTER_BACKEND == "none":
        raise ValueError("Không thể dùng --graph-mode persistent với CHECKPOINTER_BACKEND=none.")
    if args.graph_mode == "persistent" and CHECKPOINTER_BACKEND == "sqlite" and args.workers > 1:
        raise ValueError("Không chạy --workers > 1 với CHECKPOINTER_BACKEND=sqlite. Hãy dùng postgres hoặc --graph-mode stateless.")

    resolver = SubmissionRefResolver(ES_HOST, INDEX_NAME)
    prefer_applied = not args.use_relevant
    graph_app = stateless_app if args.graph_mode == "stateless" else app

    total = len(selected)
    pending_items = []
    skipped_count = 0
    for index, item in enumerate(selected, start=1):
        qid = item["id"]
        if not args.force and qid in existing_by_id:
            skipped_count += 1
            print(f"[{index}/{total}] Skip id={qid} (already in results)")
            continue
        pending_items.append((index, item))
    print(f"[Resume] selected={total} skipped={skipped_count} pending={len(pending_items)}")

    def write_results_snapshot() -> None:
        ordered_results = [
            clean_submission_result(results_by_id[q["id"]])
            for q in questions
            if q["id"] in results_by_id
        ]
        write_json_atomic(results_path, ordered_results)

    def handle_completed_future(future, index: int, item: dict[str, Any]) -> None:
        qid = item["id"]
        try:
            worker_output = future.result()
            chunk_ids = worker_output["chunk_ids"]
            relevant_docs, relevant_articles = resolver.to_submission_refs(chunk_ids)

            result = {
                "id": worker_output["id"],
                "question": worker_output["question"],
                "answer": worker_output["answer"],
                "relevant_docs": relevant_docs,
                "relevant_articles": relevant_articles,
            }
            result = clean_submission_result(result)
            diagnostics = dict(worker_output["diagnostics"])
            diagnostics["num_relevant_docs"] = len(result["relevant_docs"])
            diagnostics["num_relevant_articles"] = len(result["relevant_articles"])

            results_by_id[qid] = result
            append_jsonl(diagnostics_path, diagnostics)
            write_results_snapshot()
            print(f"[Done selected {index}/{total}] id={qid} ({diagnostics['elapsed_sec']}s)")
        except Exception as exc:
            error_record = {
                "id": qid,
                "question": item["question"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error_record)
            print(f"[ERROR] id={qid}: {exc}")
            if args.stop_on_error:
                raise

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        next_pending_idx = 0
        submitted_count = 0
        total_pending = len(pending_items)

        def submit_next() -> bool:
            nonlocal next_pending_idx, submitted_count
            if next_pending_idx >= total_pending:
                return False

            index, item = pending_items[next_pending_idx]
            next_pending_idx += 1
            submitted_count += 1
            preview = format_question_preview(item["question"])
            print(f"[Submit {submitted_count}/{total_pending} | selected {index}/{total}] id={item['id']}: {preview}")
            future = executor.submit(run_one_item, graph_app, item, prefer_applied)
            futures[future] = (index, item)
            if args.sleep > 0:
                time.sleep(args.sleep)
            return True

        for _ in range(min(args.workers, total_pending)):
            submit_next()

        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                index, item = futures.pop(future)
                try:
                    handle_completed_future(future, index, item)
                except Exception:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                submit_next()

    if not pending_items:
        write_results_snapshot()

    if not args.no_zip:
        make_submission_zip(results_path, zip_path)

    print(f"Saved results: {results_path}")
    if not args.no_zip:
        print(f"Saved zip:     {zip_path}")


if __name__ == "__main__":
    main()
