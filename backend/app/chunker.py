"""Split source files into semantically meaningful chunks.

Python is parsed with `ast` so every chunk is a whole function, class, or method.
Brace/indent languages use declaration-boundary heuristics. Everything else falls
back to overlapping line windows.
"""
import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import settings

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".hpp": "cpp", ".cs": "csharp", ".swift": "swift", ".scala": "scala", ".sh": "shell",
    ".sql": "sql", ".md": "markdown", ".rst": "rst", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".html": "html", ".css": "css", ".vue": "vue", ".svelte": "svelte",
}
SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "out", "target", "vendor", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".next", "coverage", ".idea", ".vscode",
}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "go.sum"}

# Top-level declarations in brace-style languages. Group 1 captures the name where possible.
DECL = re.compile(
    r"""^(?:
        (?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+(\w+)
      | (?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)
      | (?:export\s+)?(?:interface|type|enum)\s+(\w+)
      | (?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(|function|\w+\s*=>)
      | func\s+(?:\([^)]*\)\s*)?(\w+)
      | type\s+(\w+)\s+(?:struct|interface)
      | (?:pub(?:\(\w+\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl(?:<[^>]*>)?)\s+(\w+)
      | (?:public|private|protected|internal)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|record|enum|object)\s+(\w+)
      | def\s+(\w+)
      | module\s+(\w+)
    )""",
    re.VERBOSE,
)
LEAD_IN = re.compile(r"^\s*(//|/\*|\*|#|@|///)")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@dataclass
class Chunk:
    path: str
    language: str
    symbol: str | None
    kind: str
    start_line: int  # 1-based, inclusive
    end_line: int
    content: str

    def embedding_text(self) -> str:
        header = f"{self.path}" + (f" :: {self.symbol}" if self.symbol else "")
        return f"{header}\n{self.content}"

    def lexemes(self) -> str:
        """Text for Postgres full-text search, with identifiers expanded (getUserName -> get user name)."""
        extra = set()
        for ident in IDENT.findall(self.content) + IDENT.findall(self.symbol or ""):
            parts = CAMEL.findall(ident)
            if len(parts) > 1:
                extra.update(p.lower() for p in parts)
        path_words = re.split(r"[/._\-]", self.path)
        return " ".join([self.path, " ".join(path_words), self.symbol or "", self.content, " ".join(sorted(extra))])


def language_for(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.name in SKIP_FILES or p.name.endswith(".min.js"):
        return None
    return LANGUAGES.get(p.suffix.lower())


def chunk_file(path: str, text: str) -> list[Chunk]:
    lang = language_for(path)
    if lang is None or not text.strip():
        return []
    lines = text.splitlines()
    if lang == "python":
        try:
            return _chunk_python(path, lines, text)
        except SyntaxError:
            pass
    elif lang not in {"markdown", "rst", "toml", "yaml", "html", "css", "sql", "shell"}:
        return _chunk_by_declarations(path, lang, lines)
    return _windows(path, lang, lines, 1, len(lines), None, "window")


# ---------------------------------------------------------------- helpers

def _make(path, lang, lines, start, end, symbol, kind) -> Chunk:
    return Chunk(path, lang, symbol, kind, start, end, "\n".join(lines[start - 1:end]))


def _windows(path, lang, lines, start, end, symbol, kind) -> list[Chunk]:
    """Split [start, end] into overlapping windows of at most max_chunk_lines."""
    size, overlap = settings.max_chunk_lines, settings.window_overlap
    out, s = [], start
    while s <= end:
        e = min(s + size - 1, end)
        chunk = _make(path, lang, lines, s, e, symbol, kind)
        if chunk.content.strip():
            out.append(chunk)
        if e == end:
            break
        s = e - overlap + 1
    return out


def _span(path, lang, lines, start, end, symbol, kind) -> list[Chunk]:
    if end - start + 1 <= settings.max_chunk_lines:
        return [_make(path, lang, lines, start, end, symbol, kind)]
    return _windows(path, lang, lines, start, end, symbol, kind)


def _nonblank(lines, start, end) -> int:
    return sum(1 for line in lines[start - 1:end] if line.strip())


# ---------------------------------------------------------------- python

def _node_start(node) -> int:
    decos = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decos])


def _chunk_python(path, lines, text) -> list[Chunk]:
    tree = ast.parse(text)
    out: list[Chunk] = []
    pending_start = None  # start of a run of module-level statements

    def flush_block(end):
        nonlocal pending_start
        if pending_start is not None and _nonblank(lines, pending_start, end) >= settings.min_chunk_lines:
            out.extend(_span(path, "python", lines, pending_start, end, None, "block"))
        pending_start = None

    for node in tree.body:
        start, end = _node_start(node), node.end_lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_block(start - 1)
            out.extend(_span(path, "python", lines, start, end, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            flush_block(start - 1)
            out.extend(_chunk_python_class(path, lines, node))
        else:
            if pending_start is None:
                pending_start = start
    flush_block(len(lines))
    return out


def _chunk_python_class(path, lines, cls: ast.ClassDef) -> list[Chunk]:
    start, end = _node_start(cls), cls.end_lineno
    if end - start + 1 <= settings.max_chunk_lines:
        return [_make(path, "python", lines, start, end, cls.name, "class")]

    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out: list[Chunk] = []
    # Class header: signature, docstring, and class attributes up to the first method.
    header_end = (_node_start(methods[0]) - 1) if methods else end
    out.extend(_span(path, "python", lines, start, header_end, cls.name, "class"))
    for m in methods:
        out.extend(_span(path, "python", lines, _node_start(m), m.end_lineno, f"{cls.name}.{m.name}", "method"))
    return out


# ---------------------------------------------------------------- brace / other languages

def _chunk_by_declarations(path, lang, lines) -> list[Chunk]:
    boundaries: list[tuple[int, str | None]] = []
    for i, line in enumerate(lines, start=1):
        m = DECL.match(line)
        if not m:
            continue
        name = next((g for g in m.groups() if g), None)
        # Pull leading comments / annotations into the chunk.
        s = i
        while s > 1 and LEAD_IN.match(lines[s - 2]) and (not boundaries or s - 1 > boundaries[-1][0]):
            s -= 1
        boundaries.append((s, name))

    if not boundaries:
        return _windows(path, lang, lines, 1, len(lines), None, "window")

    segments: list[tuple[int, int, str | None]] = []
    if boundaries[0][0] > 1:
        segments.append((1, boundaries[0][0] - 1, None))
    for idx, (s, name) in enumerate(boundaries):
        e = boundaries[idx + 1][0] - 1 if idx + 1 < len(boundaries) else len(lines)
        segments.append((s, e, name))

    # Merge tiny segments (one-line type aliases, consts) into their predecessor.
    merged: list[list] = []
    for s, e, name in segments:
        if merged and _nonblank(lines, s, e) < settings.min_chunk_lines and \
                merged[-1][1] - merged[-1][0] + (e - s) < settings.max_chunk_lines:
            merged[-1][1] = e
        else:
            merged.append([s, e, name])

    out: list[Chunk] = []
    for s, e, name in merged:
        if _nonblank(lines, s, e) == 0:
            continue
        out.extend(_span(path, lang, lines, s, e, name, "function" if name else "block"))
    return out
