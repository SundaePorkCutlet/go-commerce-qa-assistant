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

GO_SYMBOL_RE = re.compile(
    r"^\s*func\s*(?:\((?P<recv>[^)]*)\)\s*)?(?P<func>[A-Za-z_][A-Za-z0-9_]*)\s*\(|"
    r"^\s*type\s+(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s+"
)


def infer_chunk_kind(rel_path: str, part: str) -> str:
    if rel_path.endswith(".go"):
        return "call_site"
    return "doc_section"


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


def build_go_symbol_chunks(rel_path: str, content: str, service: str | None) -> list[Chunk]:
    lines = content.replace("\r\n", "\n").split("\n")
    decls: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        m = GO_SYMBOL_RE.match(line)
        if not m:
            continue
        if m.group("func"):
            recv = (m.group("recv") or "").strip()
            func_name = m.group("func")
            symbol = f"{recv}.{func_name}" if recv else func_name
            kind = "method_def" if recv else "func_def"
        else:
            symbol = m.group("type") or "type"
            kind = "type_def"
        decls.append((idx, symbol, kind))

    if not decls:
        return []

    chunks: list[Chunk] = []
    for i, (start_idx, symbol, kind) in enumerate(decls):
        # Include directly attached comment block above declaration.
        chunk_start = start_idx
        j = start_idx - 1
        while j >= 0 and (lines[j].strip().startswith("//") or lines[j].strip() == ""):
            chunk_start = j
            j -= 1

        end_idx = decls[i + 1][0] if i + 1 < len(decls) else len(lines)
        symbol_block = "\n".join(lines[chunk_start:end_idx]).strip()
        if not symbol_block:
            continue

        parts = chunk_text(symbol_block, size=1800, overlap=120)
        for part_idx, part in enumerate(parts):
            symbol_hint = symbol if len(parts) == 1 else f"{symbol}#part{part_idx + 1}"
            cid = hashlib.sha1(
                f"{rel_path}:{symbol_hint}:{chunk_start + 1}:{end_idx}:{part[:64]}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    id=cid,
                    path=rel_path,
                    text=part,
                    service=service,
                    symbol_hint=symbol_hint,
                    kind=kind,
                    start_line=chunk_start + 1,
                    end_line=end_idx,
                )
            )
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
        service = detect_service(rel)

        if path.suffix.lower() == ".go":
            symbol_chunks = build_go_symbol_chunks(rel, content, service)
            if symbol_chunks:
                results.extend(symbol_chunks)
                continue

        for i, part in enumerate(chunk_text(content)):
            cid = hashlib.sha1(f"{rel}:{i}:{part[:64]}".encode("utf-8")).hexdigest()
            results.append(
                Chunk(
                    id=cid,
                    path=rel,
                    text=part,
                    service=service,
                    symbol_hint=None,
                    kind=infer_chunk_kind(rel, part),
                    start_line=None,
                    end_line=None,
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

