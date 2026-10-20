
from pydantic import BaseModel, ValidationError
from functools import cache
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from src.data_models import MinimalSource, StudentSearchResults, \
    MinimalSearchResults, RagDataset, UnansweredQuestion
from typing import Any
from pathlib import Path


import bm25s

Record = dict[str, Any]
INDEX_DIR = "data/processed/"
MIN_IOU = 0.05
MAX_CONTEXT_CHARS = 10000


def index_chunks(chunks: list[Document], index_dir: str) -> None:
    """Tokenize the chunks, build the BM25 index and save it."""
    records = [
        {"text": c.page_content,
            "file_path": c.metadata["source"],
            "first_character_index": c.metadata["start_index"],
            "last_character_index": c.metadata["end_index"]}
        for c in chunks]

    tokens = bm25s.tokenize(
        [r["text"] for r in records],
        show_progress=True, leave=False)

    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True, leave_progress=False)
    retriever.save(index_dir, corpus=records)


def load_retriever() -> bm25s.BM25 | None:
    """Load the saved index, or None if it does not exist."""
    try:
        return bm25s.BM25.load(INDEX_DIR, load_corpus=True)
    except OSError:
        print(f"Error: no index found under {INDEX_DIR}. "
              "Run the 'index' command first.")
        return None


def retrieve(queries: list[str], k: int) -> list[list[Record]] | None:
    """Return the top-k chunk records of each query, best first."""
    retriever = load_retriever()
    if retriever is None:
        return None

    progress = len(queries) > 1
    tokens = bm25s.tokenize(
        queries, show_progress=progress, leave=False
    )

    docs, _ = retriever.retrieve(
        tokens, k=min(k, len(retriever.corpus)),
        show_progress=progress, leave_progress=False
    )
    return [list(doc) for doc in docs]


def get_search(questions: list[UnansweredQuestion], k: int
               ) -> StudentSearchResults | None:
    """Retrieve the sources of every question."""
    if not questions:
        print("Error: no queries given.")
        return None

    records = retrieve([q.question for q in questions], k)
    if records is None:
        return None

    try:
        results = [
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=r["file_path"],
                        first_character_index=r["first_character_index"],
                        last_character_index=r["last_character_index"])
                    for r in chunks])
            for q, chunks in zip(questions, records)]
        return StudentSearchResults(k=k, search_results=results)
    except ValidationError as e:
        print(e)
        return None


@cache
def get_llm() -> HuggingFacePipeline:
    """Load the model once and reuse it."""
    return HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen3-0.6B", task="text-generation",
        pipeline_kwargs={"max_new_tokens": 300, "do_sample": False,
                         "return_full_text": False})


def recall(expected: list[MinimalSource],
           retrieved: list[MinimalSource]) -> float:
    """Share of expected sources hit by a retrieved source.

    A source is hit when a retrieved one is in the same file and its
    character range overlaps with an IoU of at least MIN_IOU.
    """
    def iou(a: MinimalSource, b: MinimalSource) -> float:
        if a.file_path != b.file_path:
            return 0.0
        inter = (min(a.last_character_index, b.last_character_index)
                 - max(a.first_character_index, b.first_character_index))
        if inter <= 0:
            return 0.0
        union = ((a.last_character_index - a.first_character_index)
                 + (b.last_character_index - b.first_character_index)
                 - inter)
        return inter / union

    hits = sum(any(iou(e, r) >= MIN_IOU for r in retrieved) for e in expected)
    return hits / len(expected)


def print_search_result(stud_search: StudentSearchResults) -> None:
    """Print one 'path [first:last]' line per retrieved source."""
    for result in stud_search.search_results[0].retrieved_sources:
        print(f"{result.file_path} "
              f"[{result.first_character_index}:"
              f"{result.last_character_index}]")


def save_json(
        result: BaseModel, save_directory: str, filename: str, label: str
) -> None:
    """Write a model as JSON to <save_directory>/<filename>."""
    path = Path(save_directory) / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(result.model_dump_json(indent=4))
    except OSError as e:
        print(f"Error: cannot write {path}: {e}")
        return
    print(f"Saved {label} to {path}")


def load_search_results(path: str) -> StudentSearchResults | None:
    """Load a StudentSearchResults JSON file, or None on error."""
    try:
        with open(path, "r") as f:
            return StudentSearchResults.model_validate_json(f.read())
    except (OSError, ValidationError) as e:
        print(f"Error: cannot read search results {path}: {e}")
        return None


def load_json_dataset(dataset_path: str) -> RagDataset | None:
    """Load a question dataset, answered or not, or None on error."""
    try:
        with open(dataset_path, "r") as f:
            return RagDataset.model_validate_json(f.read())
    except (OSError, ValidationError) as e:
        print(f"Error: cannot read dataset {dataset_path}: {e}")
        return None


def generate_answer(query: str, chunks: list[Record]) -> str:
    """Answer the query from the given chunks only."""
    context = "\n\n".join(
        f"file_path: '{c['file_path']}' text: '{c['text']}'"
        for c in chunks)[:MAX_CONTEXT_CHARS]

    prompt = (
        "Answer the question using only the context. Be specific: copy "
        "exact names, flags, commands and values from the context. "
        "If the context does not contain the answer, say so."
        f"\n\nContext:\n{context}\n\nQuestion: {query}"
    )

    llm = get_llm()
    chat = llm.pipeline.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return str(llm.invoke(chat)).strip()
