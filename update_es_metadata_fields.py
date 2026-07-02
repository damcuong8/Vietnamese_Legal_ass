#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

from elasticsearch import Elasticsearch, helpers


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OTHERS_METADATA = BASE_DIR / "data/raw_law/law_only/others_doc/metadata_merged_active.jsonl"
DEFAULT_LAW_METADATA = BASE_DIR / "data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl"
DEFAULT_STATE_PATH = BASE_DIR / "dbs/es_metadata_update_state.json"

DOCNO_LOCAL_RE = re.compile(r"(UBND|HĐND|HDND)", re.IGNORECASE)


def is_local_doc(issuing_authority: str, document_number: str) -> bool:
    authority = (issuing_authority or "").strip().lower()
    docno = (document_number or "").strip()
    return (
        authority.startswith("tỉnh ")
        or authority.startswith("thành phố ")
        or "ủy ban nhân dân" in authority
        or "uỷ ban nhân dân" in authority
        or "hội đồng nhân dân" in authority
        or bool(DOCNO_LOCAL_RE.search(docno))
    )


def load_metadata(others_path: Path, law_path: Path) -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}

    with others_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("id") or "").strip()
            if not doc_id:
                continue
            authority = str(row.get("issuing_authority") or "").strip()
            document_number = str(row.get("document_number") or "").strip()
            meta[doc_id] = {
                "auth": authority,
                "local": is_local_doc(authority, document_number),
            }

    with law_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("id") or "").strip()
            if not doc_id:
                continue
            attrs = row.get("thuoc_tinh") or {}
            authority = str(attrs.get("Nơi ban hành") or "").strip()
            document_number = str(attrs.get("Số hiệu") or "").strip()
            meta[doc_id] = {
                "auth": authority,
                "local": is_local_doc(authority, document_number),
            }

    return meta


def load_completed_chunks(state_path: Path) -> set[int]:
    if not state_path.exists():
        return set()
    with state_path.open(encoding="utf-8") as f:
        state = json.load(f)
    return set(int(i) for i in state.get("completed_chunks", []))


def save_completed_chunks(state_path: Path, completed: set[int], stats: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_chunks": sorted(completed),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **stats,
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(state_path)


def ensure_mapping(es: Elasticsearch, index_name: str) -> None:
    props = es.indices.get_mapping(index=index_name).body[index_name]["mappings"]["properties"]
    new_props = {}
    if "issuing_authority" not in props:
        new_props["issuing_authority"] = {"type": "keyword"}
    if "is_local" not in props:
        new_props["is_local"] = {"type": "boolean"}
    if new_props:
        es.indices.put_mapping(index=index_name, properties=new_props)
        print(f"[mapping] added {new_props}", flush=True)
    else:
        print("[mapping] fields already exist", flush=True)


def chunks(items: list[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start // chunk_size, items[start:start + chunk_size]


def update_known_docs(
    es: Elasticsearch,
    index_name: str,
    meta: dict[str, dict[str, object]],
    chunk_docs: int,
    state_path: Path,
) -> dict[str, int]:
    doc_ids = sorted(meta)
    completed = load_completed_chunks(state_path)
    total_chunks = (len(doc_ids) + chunk_docs - 1) // chunk_docs
    stats = {
        "known_doc_ids": len(doc_ids),
        "chunk_docs": chunk_docs,
        "total_chunks": total_chunks,
        "updated": 0,
        "total_matched": 0,
        "failures": 0,
    }

    script = (
        "def m = params.meta.get(ctx._source.doc_id); "
        "if (m != null) { "
        "ctx._source.issuing_authority = m.get('auth'); "
        "ctx._source.is_local = m.get('local'); "
        "}"
    )

    started = time.time()
    for chunk_idx, batch_ids in chunks(doc_ids, chunk_docs):
        if chunk_idx in completed:
            continue

        body = {
            "script": {
                "lang": "painless",
                "source": script,
                "params": {"meta": {doc_id: meta[doc_id] for doc_id in batch_ids}},
            },
            "query": {"terms": {"doc_id": batch_ids}},
        }
        response = es.options(request_timeout=3600).update_by_query(
            index=index_name,
            body=body,
            conflicts="proceed",
            refresh=False,
        ).body

        stats["updated"] += int(response.get("updated", 0))
        stats["total_matched"] += int(response.get("total", 0))
        stats["failures"] += len(response.get("failures") or [])
        completed.add(chunk_idx)
        save_completed_chunks(state_path, completed, stats)

        elapsed = time.time() - started
        print(
            f"[known {chunk_idx + 1}/{total_chunks}] "
            f"matched={response.get('total', 0)} updated={response.get('updated', 0)} "
            f"took_ms={response.get('took', 0)} elapsed_s={elapsed:.1f}",
            flush=True,
        )

    return stats


def update_unknown_docs(es: Elasticsearch, index_name: str, batch_size: int) -> dict[str, int]:
    query = {
        "query": {
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "issuing_authority"}}}},
                    {"bool": {"must_not": {"exists": {"field": "is_local"}}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["doc_id", "document_number"],
    }

    def actions():
        for hit in helpers.scan(
            es,
            index=index_name,
            query=query,
            size=batch_size,
            request_timeout=600,
            preserve_order=False,
        ):
            source = hit.get("_source") or {}
            document_number = str(source.get("document_number") or "").strip()
            yield {
                "_op_type": "update",
                "_index": index_name,
                "_id": hit["_id"],
                "doc": {
                    "issuing_authority": "",
                    "is_local": is_local_doc("", document_number),
                },
            }

    ok = failed = 0
    for success, result in helpers.streaming_bulk(
        es,
        actions(),
        chunk_size=batch_size,
        request_timeout=600,
        max_retries=5,
        initial_backoff=2,
        max_backoff=60,
        raise_on_error=False,
    ):
        if success:
            ok += 1
        else:
            failed += 1
        if (ok + failed) % 10000 == 0:
            print(f"[unknown] processed={ok + failed} ok={ok} failed={failed}", flush=True)
    return {"unknown_updated": ok, "unknown_failed": failed}


def print_final_counts(es: Elasticsearch, index_name: str) -> None:
    body = {
        "size": 0,
        "aggs": {
            "is_local": {"terms": {"field": "is_local", "size": 2}},
            "has_authority": {"filter": {"exists": {"field": "issuing_authority"}}},
            "has_is_local": {"filter": {"exists": {"field": "is_local"}}},
        },
    }
    result = es.search(index=index_name, body=body).body
    print("[final]", json.dumps(result["aggregations"], ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-host", default="http://localhost:9201")
    parser.add_argument("--index", default="legal_chunks")
    parser.add_argument("--others-metadata", type=Path, default=DEFAULT_OTHERS_METADATA)
    parser.add_argument("--law-metadata", type=Path, default=DEFAULT_LAW_METADATA)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--chunk-docs", type=int, default=5000)
    parser.add_argument("--unknown-batch-size", type=int, default=2000)
    parser.add_argument("--skip-unknown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_host}")

    ensure_mapping(es, args.index)
    meta = load_metadata(args.others_metadata, args.law_metadata)
    local_count = sum(1 for item in meta.values() if item["local"])
    print(
        f"[metadata] doc_ids={len(meta)} local={local_count} nonlocal={len(meta) - local_count}",
        flush=True,
    )

    known_stats = update_known_docs(es, args.index, meta, args.chunk_docs, args.state_path)
    print(f"[known done] {json.dumps(known_stats, ensure_ascii=False)}", flush=True)

    if not args.skip_unknown:
        unknown_stats = update_unknown_docs(es, args.index, args.unknown_batch_size)
        print(f"[unknown done] {json.dumps(unknown_stats, ensure_ascii=False)}", flush=True)

    print_final_counts(es, args.index)


if __name__ == "__main__":
    main()
