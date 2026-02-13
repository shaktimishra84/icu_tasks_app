from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src.extractors import extract_pdf_pages


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_/-]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "it",
    "if",
    "then",
    "than",
    "into",
    "over",
    "under",
    "up",
    "down",
    "within",
    "without",
    "patient",
    "patients",
}


@dataclass
class ResourceChunk:
    file_path: str
    file_name: str
    page_number: int
    chunk_index: int
    system_tag: str
    text: str


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    ]


def _weights(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    max_freq = max(counts.values())
    return {
        token: (0.5 + 0.5 * (count / max_freq)) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size_words - overlap_words)
    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words).strip())
        start += step
    return chunks


class KnowledgeBase:
    def __init__(self, resources_root: Path, store_path: Path) -> None:
        self.resources_root = Path(resources_root)
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[ResourceChunk] = []
        self._last_indexed_files: list[str] = []

    def build_from_resources(self) -> None:
        systems_root = self.resources_root / "all systems"
        index_root = systems_root if systems_root.exists() else self.resources_root
        pdf_paths = sorted(path for path in index_root.rglob("*.pdf") if path.is_file())
        chunks: list[ResourceChunk] = []

        for pdf_path in pdf_paths:
            try:
                relative_path = pdf_path.relative_to(index_root).as_posix()
            except ValueError:
                relative_path = pdf_path.as_posix()

            parts = relative_path.split("/")
            system_tag = parts[0] if len(parts) > 1 else "unclassified"
            data = pdf_path.read_bytes()
            pages = extract_pdf_pages(data)

            for page_number, page_text in pages:
                for chunk_index, chunk_text in enumerate(_chunk_text(page_text), start=1):
                    chunks.append(
                        ResourceChunk(
                            file_path=relative_path,
                            file_name=pdf_path.name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            system_tag=system_tag,
                            text=chunk_text,
                        )
                    )

        self._chunks = chunks
        self._last_indexed_files = [chunk.file_path for chunk in chunks]
        self.save()

    def save(self) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self._chunks],
            "files_indexed": sorted(set(self._last_indexed_files)),
        }
        with self.store_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def load_from_store(self) -> bool:
        if not self.store_path.exists():
            return False
        try:
            with self.store_path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except Exception:
            return False

        chunks_raw = payload.get("chunks", []) if isinstance(payload, dict) else []
        loaded_chunks: list[ResourceChunk] = []
        for raw in chunks_raw:
            if not isinstance(raw, dict):
                continue
            try:
                loaded_chunks.append(
                    ResourceChunk(
                        file_path=str(raw.get("file_path", "")),
                        file_name=str(raw.get("file_name", "")),
                        page_number=int(raw.get("page_number", 0)),
                        chunk_index=int(raw.get("chunk_index", 0)),
                        system_tag=str(raw.get("system_tag", "unclassified")),
                        text=str(raw.get("text", "")),
                    )
                )
            except Exception:
                continue
        self._chunks = loaded_chunks
        self._last_indexed_files = [chunk.file_path for chunk in loaded_chunks]
        return bool(loaded_chunks)

    def chunk_count(self) -> int:
        return len(self._chunks)

    def file_count(self) -> int:
        return len(set(chunk.file_path for chunk in self._chunks))

    def list_files(self) -> list[str]:
        return sorted(set(chunk.file_path for chunk in self._chunks))

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        only_neuro: bool = True,
        system_tag: str | None = None,
    ) -> list[tuple[ResourceChunk, float]]:
        if not self._chunks:
            return []

        filtered = [
            chunk
            for chunk in self._chunks
            if (
                (not only_neuro or chunk.system_tag == "01_neuro")
                and (system_tag is None or chunk.system_tag == system_tag)
            )
        ]
        if not filtered:
            return []

        documents_tokens = [_tokenize(chunk.text) for chunk in filtered]
        doc_freq: Counter[str] = Counter()
        for token_list in documents_tokens:
            doc_freq.update(set(token_list))

        total_docs = len(filtered)
        idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
            for token, frequency in doc_freq.items()
        }

        query_tokens = _tokenize(query)
        query_weights = _weights(query_tokens, idf)

        scored_results: list[tuple[ResourceChunk, float]] = []
        for chunk, token_list in zip(filtered, documents_tokens):
            chunk_weights = _weights(token_list, idf)
            score = _cosine_similarity(query_weights, chunk_weights)
            if score > 0:
                scored_results.append((chunk, score))

        scored_results.sort(key=lambda row: row[1], reverse=True)
        return scored_results[: max(1, top_k)]
