import re
import os

from collections.abc import Iterator
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as RCTS, Language)
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

# Only allow .py, .txt, .md (plus Makefile / Dockerfile). No hidden files.
ALLOWED_FILES = re.compile(
    r"""
    (?!\.)
    (
    .+\.(py|txt|md)
    | Makefile | Dockerfile
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Supported Languages.
LANG = {".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".js": Language.JS,
        ".txt": Language.MARKDOWN}


def _iter_files(root: Path, extensions: set[str] | None = None
                ) -> Iterator[Path]:
    """Yield every allowed file under root, skipping ignored folders."""
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
    """Read every allowed file under raw_data_path and split it in chunks.

    Python files and Markdown/text files each use their own splitter.
    Each chunk keeps its file path and character span in its metadata.
    """
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

    # Listing the files for the progress bar.
    files = [p for p in _iter_files(root=root)
             if p.is_file() and ".git" not in p.parts]

    for p in tqdm(files, desc="Chunking", unit="file", leave=False):
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # Print without breaking the progress bar.
            tqdm.write(f"Skipping {p}: {e}")
            continue

        rel = p.relative_to(root)
        for c in split.get(p.suffix, generic).create_documents(
                [text], [{"source": str(p)}]):
            c.metadata["end_index"] = (
                c.metadata["start_index"] + len(c.page_content))

            c.page_content = f"File: {rel}\n{c.page_content}"
            chunks.append(c)
    tqdm.close(p)
    return chunks
