from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

from pqa.models import Chunk


INCLUDE_SUFFIXES = {".go", ".md", ".yaml", ".yml", ".json"}
EXCLUDE_PATTERNS = [
    re.compile(r"^\.git/"),
    re.compile(r"^\.cursor/"),
    re.compile(r"^\.gopath/"),
    re.compile(r"^tools/project-qa-assistant/"),
    re.compile(r"/node_modules/"),
    re.compile(r"/vendor/"),
    re.compile(r"^tools/project-qa-assistant/data/"),
    re.compile(r"INTERVIEW_PREP"),
    re.compile(r"agent-transcripts"),
]


def should_index(rel_path: str) -> bool:
    if any(p.search(rel_path) for p in EXCLUDE_PATTERNS):
        return False
    suffix = Path(rel_path).suffix.lower()
    return suffix in INCLUDE_SUFFIXES


def detect_service(rel_path: str) -> str | None:
    for name in ("ORDERFC", "PAYMENTFC", "PRODUCTFC", "USERFC", "frontend"):
        if rel_path.startswith(name + "/"):
            return name
    return None


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    if len(normalized) <= size:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(repo_root: Path) -> list[Chunk]:
    results: list[Chunk] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_root))
        if not should_index(rel):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, part in enumerate(chunk_text(content)):
            cid = hashlib.sha1(f"{rel}:{i}:{part[:64]}".encode("utf-8")).hexdigest()
            results.append(
                Chunk(
                    id=cid,
                    path=rel,
                    text=part,
                    service=detect_service(rel),
                    symbol_hint=None,
                )
            )
    return results


def write_index(chunks: list[Chunk], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def read_index(index_path: Path) -> list[Chunk]:
    if not index_path.exists():
        return []
    results: list[Chunk] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(Chunk.model_validate_json(line))
    return results

