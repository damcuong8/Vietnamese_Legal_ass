from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_RESULTS = BASE_DIR / "submission_eval_merged_article_clean_dedup_no_vbhn_v2" / "results.json"
DEFAULT_NO_LOCAL_RESULTS = BASE_DIR / "submission_eval_merged_article_clean_dedup_no_vbhn_no_local" / "results.json"
DEFAULT_RERUN_RESULTS = BASE_DIR / "rerun_empty_after_no_local" / "results.json"
DEFAULT_OUT_DIR = BASE_DIR / "submission_eval_merged_article_clean_dedup_no_vbhn_no_local_central_repair"

DOCNO_RE = re.compile(
    r"\b\d{1,4}/\d{4}/[A-ZĐÂĂÊÔƠƯa-zđâăêôơư0-9./-]+(?:-[A-ZĐÂĂÊÔƠƯa-zđâăêôơư0-9]+)*\b"
)
ARTICLE_DETAIL_RE = re.compile(r"\s*\([^)]*\)")
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+")

SOURCE_FIELDS = [
    "chunk_id",
    "document_number",
    "raw_title",
    "article_no",
    "article_title",
    "raw_content",
]

STOPWORDS = {
    "anh",
    "bao",
    "bi",
    "cac",
    "can",
    "cho",
    "co",
    "cua",
    "duoc",
    "hay",
    "hoi",
    "khac",
    "khi",
    "khong",
    "la",
    "lam",
    "mot",
    "nay",
    "neu",
    "nguoi",
    "nhu",
    "phai",
    "qua",
    "sau",
    "tai",
    "the",
    "thi",
    "theo",
    "toi",
    "trong",
    "tu",
    "va",
    "ve",
    "voi",
}


def read_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_zip(results_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return text.replace("Đ", "D").replace("đ", "d")


def norm(text: str) -> str:
    return strip_accents(text).upper()


def parse_ref(ref: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in str(ref or "").split("|")]
    document_number = parts[0] if len(parts) > 0 else ""
    title = parts[1] if len(parts) > 1 else ""
    article = parts[2] if len(parts) > 2 else ""
    return document_number, title, article


def clean_article_no(article_no: str) -> str:
    return ARTICLE_DETAIL_RE.sub("", str(article_no or "")).strip()


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def is_local_ref(document_number: str, title: str = "") -> bool:
    number_norm = norm(document_number)
    title_norm = norm(title)
    local_number_markers = (
        "QD-UBND",
        "QÐ-UBND",
        "NQ-HDND",
        "QD-HDND",
        "CT-UBND",
        "CT-CTUBND",
        "CTUBND",
        "UBND",
        "HDND",
    )
    if any(marker in number_norm for marker in local_number_markers):
        return True

    local_title_markers = (
        "UY BAN NHAN DAN",
        "HOI DONG NHAN DAN",
        "UBND",
        "HDND",
        "SO LAO DONG",
        "SO NOI VU",
        "SO Y TE",
        "SO GIAO DUC",
        "SO KE HOACH",
        "TINH ",
        "THANH PHO ",
    )
    return any(marker in title_norm for marker in local_title_markers)


def is_central_doc_number(document_number: str) -> bool:
    number_norm = norm(document_number)
    if is_local_ref(document_number):
        return False
    central_markers = (
        "ND-CP",
        "/QH",
        "TT-",
        "TTLT-",
        "QD-TTG",
        "NQ-CP",
        "NQ-HDTP",
        "UBTVQH",
        "PL-UBTVQH",
    )
    return any(marker in number_norm for marker in central_markers)


def tokenize(text: str) -> set[str]:
    normalized = strip_accents(text).lower()
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) >= 3 and token not in STOPWORDS
    }


def extract_doc_numbers(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for match in DOCNO_RE.findall(str(text or "")):
            docno = match.strip(".,;:()[]{}\"'”")
            if is_central_doc_number(docno):
                found.append(docno)
    return unique_keep_order(found)


def build_evidence_text(*items: dict[str, Any] | None) -> str:
    parts: list[str] = []
    for item in items:
        if not item:
            continue
        parts.append(str(item.get("question") or ""))
        parts.append(str(item.get("answer") or ""))
        parts.extend(str(ref) for ref in item.get("relevant_docs") or [])
        parts.extend(str(ref) for ref in item.get("relevant_articles") or [])
    return "\n".join(part for part in parts if part)


class CentralRefRepairer:
    def __init__(self, es_host: str, index_name: str, max_doc_chunks: int):
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        self.max_doc_chunks = max_doc_chunks
        self._doc_cache: dict[str, list[dict[str, Any]]] = {}

    def fetch_doc_chunks(self, document_number: str) -> list[dict[str, Any]]:
        if document_number in self._doc_cache:
            return self._doc_cache[document_number]

        body = {
            "query": {"term": {"document_number": document_number}},
            "size": self.max_doc_chunks,
            "_source": SOURCE_FIELDS,
        }
        response = self.es.search(index=self.index_name, body=body)
        chunks = [
            hit.get("_source") or {}
            for hit in response.get("hits", {}).get("hits", [])
            if hit.get("_source")
        ]
        self._doc_cache[document_number] = chunks
        return chunks

    def fetch_ref_texts(self, refs: list[str]) -> str:
        texts: list[str] = []
        for ref in refs:
            document_number, _title, article_no = parse_ref(ref)
            if not document_number:
                continue
            chunks = self.fetch_doc_chunks(document_number)
            if article_no:
                article_norm = norm(clean_article_no(article_no))
                chunks = [
                    src
                    for src in chunks
                    if norm(clean_article_no(str(src.get("article_no") or ""))) == article_norm
                ] or chunks
            for src in chunks[:20]:
                texts.append(str(src.get("raw_title") or ""))
                texts.append(str(src.get("article_title") or ""))
                texts.append(str(src.get("raw_content") or ""))
        return "\n".join(text for text in texts if text)

    def score_chunk(self, src: dict[str, Any], question: str, evidence_text: str) -> float:
        article_title = str(src.get("article_title") or "")
        raw_content = str(src.get("raw_content") or "")
        article_no = clean_article_no(str(src.get("article_no") or ""))
        title_norm = norm(article_title)
        content_norm = norm(raw_content)
        combined_norm = f"{title_norm}\n{content_norm}"

        q_tokens = tokenize(question)
        title_tokens = tokenize(article_title)
        content_tokens = tokenize(raw_content)

        score = 0.0
        score += len(q_tokens & title_tokens) * 4.0
        score += min(len(q_tokens & content_tokens), 30) * 1.2

        question_norm = norm(question)
        if "CAP GIAY PHEP" in question_norm and "CAP GIAY PHEP" in combined_norm:
            score += 18.0
        if "THU TUC" in question_norm and "THU TUC" in combined_norm:
            score += 8.0
        if any(term in question_norm for term in ("THOI GIAN", "THOI HAN", "BAO LAU", "NGAY")):
            if "THOI HAN" in combined_norm:
                score += 14.0
            if "NGAY LAM VIEC" in combined_norm:
                score += 12.0
        if "TRINH TU" in title_norm and "THU TUC" in title_norm:
            score += 10.0
        if "HO SO" in title_norm and "HO SO" in question_norm:
            score += 8.0
        if article_no and norm(article_no) in norm(evidence_text):
            score += 25.0

        if "HIEU LUC THI HANH" in title_norm:
            score -= 12.0
        if "TRACH NHIEM" in title_norm and "TRACH NHIEM" not in question_norm:
            score -= 8.0
        if title_norm.startswith("PHU LUC") and "PHU LUC" not in question_norm:
            score -= 4.0

        return score

    def central_refs_for_item(
        self,
        question: str,
        evidence_text: str,
        local_refs: list[str],
        top_docs: int,
        top_articles: int,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        local_ref_text = self.fetch_ref_texts(local_refs)
        direct_doc_numbers = extract_doc_numbers(evidence_text)
        local_doc_numbers = [
            docno
            for docno in extract_doc_numbers(local_ref_text)
            if docno not in set(direct_doc_numbers)
        ]
        doc_numbers = direct_doc_numbers or local_doc_numbers

        used_local_fallback = False
        candidates: list[tuple[float, dict[str, Any]]] = []

        def add_candidates(numbers: list[str]) -> None:
            for document_number in numbers:
                for src in self.fetch_doc_chunks(document_number):
                    docno = str(src.get("document_number") or "").strip()
                    title = str(src.get("raw_title") or "").strip()
                    if not docno or not title or is_local_ref(docno, title):
                        continue
                    score = self.score_chunk(src, question, evidence_text)
                    candidates.append((score, src))

        add_candidates(doc_numbers)
        if not candidates and direct_doc_numbers and local_doc_numbers:
            doc_numbers = local_doc_numbers
            used_local_fallback = True
            add_candidates(doc_numbers)

        candidates.sort(
            key=lambda pair: (
                pair[0],
                norm(str(pair[1].get("document_number") or "")) in {norm(x) for x in doc_numbers},
            ),
            reverse=True,
        )

        relevant_docs: list[str] = []
        relevant_articles: list[str] = []
        for _score, src in candidates:
            document_number = str(src.get("document_number") or "").strip()
            title = str(src.get("raw_title") or "").strip()
            article_no = clean_article_no(str(src.get("article_no") or ""))
            if not document_number or not title:
                continue

            relevant_docs.append(f"{document_number}|{title}")
            if article_no:
                relevant_articles.append(f"{document_number}|{title}|{article_no}")

            relevant_docs = unique_keep_order(relevant_docs)
            relevant_articles = unique_keep_order(relevant_articles)
            if len(relevant_docs) >= top_docs and len(relevant_articles) >= top_articles:
                break

        relevant_docs = relevant_docs[:top_docs]
        relevant_articles = relevant_articles[:top_articles]
        details = {
            "direct_central_doc_numbers": direct_doc_numbers,
            "local_ref_central_doc_numbers": local_doc_numbers,
            "central_doc_numbers": doc_numbers,
            "used_local_fallback": used_local_fallback,
            "num_candidates": len(candidates),
            "selected_docs": relevant_docs,
            "selected_articles": relevant_articles,
        }
        return relevant_docs, relevant_articles, details


def repair_results(args: argparse.Namespace) -> dict[str, Any]:
    source_results = read_json_array(args.source_results)
    no_local_results = read_json_array(args.no_local_results)
    rerun_results = read_json_array(args.rerun_results) if args.rerun_results.exists() else []

    source_by_id = {item.get("id"): item for item in source_results}
    rerun_by_id = {item.get("id"): item for item in rerun_results}
    repairer = CentralRefRepairer(args.es_host, args.index_name, args.max_doc_chunks)

    repaired_results: list[dict[str, Any]] = []
    repaired_ids: list[Any] = []
    local_filtered_ids: list[Any] = []
    unresolved_ids: list[Any] = []
    repair_details: dict[str, Any] = {}

    for item in no_local_results:
        new_item = dict(item)
        item_id = new_item.get("id")
        original_docs = list(new_item.get("relevant_docs") or [])
        original_articles = list(new_item.get("relevant_articles") or [])
        docs = [
            ref
            for ref in original_docs
            if not is_local_ref(*parse_ref(ref)[:2])
        ]
        articles = [
            ref
            for ref in original_articles
            if not is_local_ref(*parse_ref(ref)[:2])
        ]
        if len(docs) != len(original_docs) or len(articles) != len(original_articles):
            local_filtered_ids.append(item_id)
            new_item["relevant_docs"] = docs
            new_item["relevant_articles"] = articles
        needs_repair = not docs and not articles

        if needs_repair:
            source_item = source_by_id.get(item_id)
            rerun_item = rerun_by_id.get(item_id)
            question = str(new_item.get("question") or (source_item or {}).get("question") or "")
            evidence_text = build_evidence_text(new_item, source_item, rerun_item)
            local_refs = []
            if source_item:
                local_refs.extend(source_item.get("relevant_docs") or [])
                local_refs.extend(source_item.get("relevant_articles") or [])
            if rerun_item:
                local_refs.extend(rerun_item.get("relevant_docs") or [])
                local_refs.extend(rerun_item.get("relevant_articles") or [])
            local_refs = [
                ref
                for ref in unique_keep_order([str(ref) for ref in local_refs])
                if is_local_ref(*parse_ref(ref)[:2])
            ]

            repaired_docs, repaired_articles, details = repairer.central_refs_for_item(
                question=question,
                evidence_text=evidence_text,
                local_refs=local_refs,
                top_docs=args.top_docs,
                top_articles=args.top_articles,
            )
            repair_details[str(item_id)] = details
            if repaired_docs or repaired_articles:
                new_item["relevant_docs"] = repaired_docs
                new_item["relevant_articles"] = repaired_articles
                repaired_ids.append(item_id)
            else:
                unresolved_ids.append(item_id)

        repaired_results.append(new_item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.json"
    manifest_path = args.out_dir / "central_repair_manifest.json"
    zip_path = args.out_dir / "submission.zip"

    write_json(results_path, repaired_results)
    make_zip(results_path, zip_path)
    manifest = {
        "source_results": str(args.source_results),
        "no_local_results": str(args.no_local_results),
        "rerun_results": str(args.rerun_results),
        "output_results": str(results_path),
        "output_zip": str(zip_path),
        "total": len(repaired_results),
        "local_filtered_count": len(local_filtered_ids),
        "local_filtered_ids": local_filtered_ids,
        "repaired_count": len(repaired_ids),
        "repaired_ids": repaired_ids,
        "unresolved_count": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "repair_details": repair_details,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair empty no-local submission refs by retrieving cited central legal documents."
    )
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--no-local-results", type=Path, default=DEFAULT_NO_LOCAL_RESULTS)
    parser.add_argument("--rerun-results", type=Path, default=DEFAULT_RERUN_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--es-host", default="http://localhost:9201")
    parser.add_argument("--index-name", default="legal_chunks")
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--top-articles", type=int, default=5)
    parser.add_argument("--max-doc-chunks", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for attr in ("source_results", "no_local_results", "rerun_results", "out_dir"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, BASE_DIR / path)

    manifest = repair_results(args)
    print(f"Saved results: {manifest['output_results']}")
    print(f"Saved zip: {manifest['output_zip']}")
    print(f"Repaired {manifest['repaired_count']} empty items")
    if manifest["unresolved_ids"]:
        print(f"Unresolved ids: {manifest['unresolved_ids']}")


if __name__ == "__main__":
    main()
