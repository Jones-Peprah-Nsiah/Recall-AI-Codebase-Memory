from app.chunker import Chunk, chunk_file
from app.search import to_tsquery

PY = '''
import os

CONSTANT = 1
OTHER = 2
THIRD = 3
FOURTH = 4


@decorator
def top_level(a, b):
    """Add things."""
    return a + b


class Small:
    def method(self):
        return 1
'''


def test_python_functions_and_classes_are_whole_chunks():
    chunks = chunk_file("pkg/mod.py", PY)
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["top_level"].kind == "function"
    assert by_symbol["top_level"].content.startswith("@decorator")
    assert by_symbol["Small"].kind == "class"
    assert any(c.kind == "block" and "CONSTANT" in c.content for c in chunks)


def test_large_python_class_is_split_into_methods():
    body = "\n".join(f"    def m{i}(self):\n" + "\n".join(f"        x = {j}" for j in range(8)) for i in range(12))
    src = f"class Big:\n    '''Doc.'''\n\n{body}\n"
    chunks = chunk_file("big.py", src)
    symbols = [c.symbol for c in chunks]
    assert symbols[0] == "Big"
    assert "Big.m0" in symbols and "Big.m11" in symbols
    assert all(c.end_line - c.start_line < 80 for c in chunks)


def test_line_numbers_match_content():
    src = PY
    lines = src.splitlines()
    for c in chunk_file("pkg/mod.py", src):
        assert c.content == "\n".join(lines[c.start_line - 1:c.end_line])


def test_typescript_declarations():
    src = """import x from 'y';

// Adds numbers
export function add(a: number, b: number) {
  const s = a + b;
  return s;
}

export const useThing = (id: string) => {
  const [v, setV] = useState(null);
  useEffect(() => {}, [id]);
  return v;
};

export class Store {
  items = [];
  get(i) { return this.items[i]; }
  set(i, v) { this.items[i] = v; }
}
"""
    chunks = chunk_file("src/a.ts", src)
    names = [c.symbol for c in chunks]
    assert "add" in names and "useThing" in names and "Store" in names
    add = next(c for c in chunks if c.symbol == "add")
    assert add.content.startswith("// Adds numbers")


def test_unknown_and_lockfiles_skipped():
    assert chunk_file("image.png", "xx") == []
    assert chunk_file("package-lock.json", "{}") == []


def test_markdown_windows_overlap():
    src = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_file("README.md", src)
    assert len(chunks) == 3
    assert chunks[1].start_line == chunks[0].end_line - 9


def test_lexemes_split_identifiers():
    c = Chunk("a/b.ts", "typescript", "getUserName", "function", 1, 1, "function getUserName() {}")
    lex = c.lexemes()
    assert "user" in lex and "name" in lex


def test_tsquery_sanitizes_and_splits():
    assert to_tsquery("how do we parseJSONBody?") == "body:* | json:* | parse:* | parsejsonbody:*"
    assert to_tsquery("'; drop table --") == "drop:* | table:*"
