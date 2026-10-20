from src.loader import load_and_split_files
from src.type_checker import cli_safe, PositiveInt
from src.data_models import MinimalSource, StudentSearchResults, \
    MinimalSearchResults, RagDataset, UnansweredQuestion, MinimalAnswer, \
    StudentSearchResultsAndAnswer

from pydantic import BaseModel, ValidationError
from functools import cache
from pathlib import Path
from tqdm import tqdm
from typing import Any

from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document

import fire
import bm25s


INDEX_DIR = "data/processed/"
RAW_DIR = "data/raw/"
MAX_CONTEXT_CHARS = 10000

Record = dict[str, Any]


class Rag:
    """Command-line entry point of the RAG pipeline.

    Commands and their arguments:
        index [--max_chunk_size <int>]
        search <query> [--k <int>]
        search_dataset --dataset_path <path> --save_directory <dir>
                       [--k <int>]
        answer <query> [--k <int>]
        answer_dataset --student_search_results_path <path>
                       --save_directory <dir>

    Run 'COMMAND --help' for details on one command.
    """

    @classmethod
    @cli_safe
    def index(cls, max_chunk_size: PositiveInt = 2000) -> None:
        """Chunk data/raw/ and build the BM25 index.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters.
        """
        chunk_overlap = int(max_chunk_size * 0.1)
        chunks = load_and_split_files(RAW_DIR, max_chunk_size, chunk_overlap)

        if not chunks:
            print("Error: Either path is empty or permission problem. "
                  "Could not load any files.")
            return

        cls._index_chunks(chunks, INDEX_DIR)
        print(f"Ingestion complete! Indices saved under {INDEX_DIR}")

    @staticmethod
    def _index_chunks(chunks: list[Document], index_dir: str) -> None:
        """Tokenize the chunks, build the BM25 index and save it."""
        records = [
            {"text": c.page_content,
             "file_path": c.metadata["source"],
             "first_character_index": c.metadata["start_index"],
             "last_character_index": c.metadata["end_index"]
             } for c in chunks]

        tokens = bm25s.tokenize(
            [r["text"] for r in records],
            show_progress=True, leave=False)

        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=True, leave_progress=False)
        retriever.save(index_dir, corpus=records)

    @staticmethod
    def _load_retriever() -> bm25s.BM25 | None:
        """Load the saved index, or None if it does not exist."""
        try:
            return bm25s.BM25.load(INDEX_DIR, load_corpus=True)
        except OSError:
            print(f"Error: no index found under {INDEX_DIR}. "
                  "Run the 'index' command first.")
            return None

    @classmethod
    def _retrieve(cls, queries: list[str], k: int
                  ) -> list[list[Record]] | None:
        """Return the top-k chunk records of each query, best first."""
        retriever = cls._load_retriever()
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

    @classmethod
    def _get_search(cls, questions: list[UnansweredQuestion], k: int
                    ) -> StudentSearchResults | None:
        """Retrieve the sources of every question."""
        if not questions:
            print("Error: no queries given.")
            return None

        records = cls._retrieve([q.question for q in questions], k)
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

    @classmethod
    @cli_safe
    def search(cls, query: str, k: PositiveInt = 5) -> None:
        """Print the top-k source locations of a single query.

        Args:
            query: Question to search for.
            k: Number of sources to return.
        """
        if not query.strip():
            print("Error: the query is empty.")
            return

        results = cls._get_search([UnansweredQuestion(question=query)], k)
        if results:
            cls._print_search_result(results)

    @classmethod
    @cli_safe
    def search_dataset(
            cls, dataset_path: str, save_directory: str, k: PositiveInt = 5
    ) -> None:
        """Search a whole dataset and save the StudentSearchResults JSON.

        Args:
            dataset_path: JSON file holding the questions.
            save_directory: Directory where the result file is written.
            k: Number of sources to return per question.
        """
        dataset = cls._load_json_dataset(dataset_path)
        if dataset is None:
            return

        results = cls._get_search(dataset.rag_questions, k)
        if results is None:
            return

        cls._save_json(results, save_directory, Path(dataset_path).name,
                       "student_search_results")

    @staticmethod
    @cache
    def _llm() -> HuggingFacePipeline:
        """Load the model once and reuse it."""
        return HuggingFacePipeline.from_model_id(
            model_id="Qwen/Qwen3-0.6B", task="text-generation",
            pipeline_kwargs={"max_new_tokens": 300, "do_sample": False,
                             "return_full_text": False})

    @classmethod
    def _generate_answer(cls, query: str, chunks: list[Record]) -> str:
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

        llm = cls._llm()
        chat = llm.pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return str(llm.invoke(chat)).strip()

    @classmethod
    @cli_safe
    def answer(cls, query: str, k: PositiveInt = 5) -> None:
        """Print the answer to a single query.

        Args:
            query: Question to answer.
            k: Number of sources used as context.
        """
        if not query.strip():
            print("Error: the query is empty.")
            return

        records = cls._retrieve([query], k)
        if records is None:
            return

        print(cls._generate_answer(query, records[0]))

    @classmethod
    @cli_safe
    def answer_dataset(
            cls, student_search_results_path: str, save_directory: str
    ) -> None:
        """Answer every question of a search-results file and save the JSON.

        Args:
            student_search_results_path: JSON file made by search_dataset.
            save_directory: Directory where the result file is written.
        """
        try:
            with open(student_search_results_path, "r") as f:
                search_results = StudentSearchResults.model_validate_json(
                    f.read())
        except (OSError, ValidationError) as e:
            print(f"Error: cannot read {student_search_results_path}: {e}")
            return

        retriever = cls._load_retriever()
        if retriever is None:
            return

        chunk_by_span = {
            (r["file_path"], r["first_character_index"],
             r["last_character_index"]): r
            for r in retriever.corpus}

        questions = search_results.search_results
        print(f"Loaded {len(questions)} questions")
        print("Loading model...")
        cls._llm()

        answered = []
        for result in tqdm(questions, desc="Answering", unit="question"):
            spans = [(s.file_path, s.first_character_index,
                      s.last_character_index)
                     for s in result.retrieved_sources]
            chunks = [chunk_by_span[s] for s in spans if s in chunk_by_span]
            answered.append(MinimalAnswer(
                **result.model_dump(),
                answer=cls._generate_answer(result.question, chunks)))

        output = StudentSearchResultsAndAnswer(
            k=search_results.k, search_results=answered)
        cls._save_json(output, save_directory,
                       Path(student_search_results_path).name,
                       "student_search_results_and_answer")

    @staticmethod
    def _print_search_result(stud_search: StudentSearchResults) -> None:
        """Print one 'path [first:last]' line per retrieved source."""
        for result in stud_search.search_results[0].retrieved_sources:
            print(f"{result.file_path} "
                  f"[{result.first_character_index}:"
                  f"{result.last_character_index}]")

    @staticmethod
    def _save_json(
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

    @staticmethod
    def _load_json_dataset(dataset_path: str) -> RagDataset | None:
        """Load a question dataset, answered or not, or None on error."""
        try:
            with open(dataset_path, "r") as f:
                return RagDataset.model_validate_json(f.read())
        except (OSError, ValidationError) as e:
            print(f"Error: cannot read dataset {dataset_path}: {e}")
            return None


if __name__ == "__main__":
    fire.Fire(Rag, name="rag")