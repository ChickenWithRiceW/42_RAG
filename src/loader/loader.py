import re
import os
from collections.abc import Iterator
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS, Language

# Dirs we ignore.
IGNORE_DIRS = re.compile(
    r"""
    \..*
    | __pycache__
    | node_modules
    | venv | env
    | build | dist
    | site-packages
    | .*\.egg-info
    """,
    re.VERBOSE)

# TODO: Only allow .py, .txt, .md
IGNORE_FILES = re.compile(
    r"""
    \..*                    # hidden files: .env, .DS_Store, .gitignore
    | .*~                   # editor backups: notes.md~
    | .*\.(swp|swo|tmp|bak|log|lock)
    | .*\.py[cod]           # compiled Python: .pyc, .pyo, .pyd
    | .*\.(so|dll|dylib|exe|o|a)
    | .*\.(zip|tar|gz|bz2|7z|rar)
    | .*\.(png|jpe?g|gif|bmp|ico|svg|webp|mp[34]|mov|avi)
    | Thumbs\.db
    | package-lock\.json | yarn\.lock | poetry\.lock
    | CMakeLists.txt
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Supported Languages.
LANG = {".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".js": Language.JS,
        ".txt": Language.MARKDOWN}


def _iter_files(root: Path, extensions: set[str] | None = None) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Folder filtering.
        dirnames[:] = [d for d in dirnames if not IGNORE_DIRS.fullmatch(d)]

        for name in filenames:
            # Ignore hidden files
            if IGNORE_FILES.fullmatch(name):
                continue
            path = Path(dirpath) / name
            if extensions and path.suffix.lower() not in extensions:
                continue
            yield path


def load_and_split_files(
    raw_data_path: str, max_chunk_size: int,
        chunk_overlap: int) -> None:

    cfg = dict(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    split = {
        ext: RCTS.from_language(lang, **cfg)
        for ext, lang in LANG.items()
    }

    generic = RCTS(**cfg)
    root = Path(raw_data_path)
    chunks = []

    for p in _iter_files(root=root):
        if not p.is_file() or ".git" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # binary / unreadable
            continue
        rel = p.relative_to(root)
        for c in split.get(p.suffix, generic).create_documents(
                [text], [{"source": str(p)}]):
            c.page_content = f"File: {rel}\n{c.page_content}"
            chunks.append(c)
    return chunks