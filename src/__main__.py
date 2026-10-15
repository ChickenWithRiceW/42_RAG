import typer
import functools
import sys
from typing import Annotated
from langchain_community.document_loaders import 

INDEX_DIR = "data/processed"
RAW_DIR = "data/raw"
app = typer.Typer(no_args_is_help=True)

# ----- Type hints for CLI -----
PositiveInt = Annotated[int, typer.Option(min=1)]
NoneNegativeInt = Annotated[int, typer.Option(min=0)]


@app.command()
def other():
    print("other")


@app.command()
def index(max_chunk_size: PositiveInt = 2000, raw_data: str = RAW_DIR,
            index_dir: str = INDEX_DIR, chunk_overlap: NoneNegativeInt = 200) -> None:
    """--max-chunk-size <max chunk size> --raw-data <raw data path>
    --index-dir <path to index> --chunk-overlap <size of overlap>
    """
    print("Hey!", max_chunk_size)


if __name__ == "__main__":
    app()
