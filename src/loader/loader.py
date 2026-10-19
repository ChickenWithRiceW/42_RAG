import re
import os
from collections.abc import Iterator
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS, Language
from langchain_core.documents import Document

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
ALLOWED_FILES = re.compile(
    r"""
    (?!\.)                      # never hidden: .env, .gitignore, .DS_Store
    (
        .+\.(py|txt|md)         # any .py, .txt or .md file (CMakeLists.txt included)
      | Makefile | Dockerfile   # extra exact names, optional
    )
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
            if not ALLOWED_FILES.fullmatch(name):
                continue
            path = Path(dirpath) / name
            if extensions and path.suffix.lower() not in extensions:
                continue
            yield path


def load_and_split_files(
    raw_data_path: str, max_chunk_size: int,
        chunk_overlap: int) -> list[Document] | None:

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
        except OSError as e:
            print(e)
            return None

        except UnicodeDecodeError as e:
            print(e)
            continue

        rel = p.relative_to(root)
        for c in split.get(p.suffix, generic).create_documents(
                [text], [{"source": str(p)}]):
            c.metadata["end_index"] = (c.metadata["start_index"] + len(c.page_content))
            c.page_content = f"File: {rel}\n{c.page_content}"
            chunks.append(c)
    return chunks
