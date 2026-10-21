from src.loader import load_and_split_files
from src.type_checker import cli_safe, PositiveInt
from src.data_models import UnansweredQuestion, AnsweredQuestion, \
    MinimalAnswer, StudentSearchResultsAndAnswer

from src.helper import save_json, load_search_results, \
    load_json_dataset, index_chunks, get_search, generate_answer, \
    print_search_result, retrieve, load_retriever, get_llm, recall

from pathlib import Path
from tqdm import tqdm

import fire


INDEX_DIR = "data/processed/"
RAW_DIR = "data/raw/"
MAX_SOURCE_LENGTH = 2000
RECALL_KS = (1, 3, 5, 10)


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
        evaluate --student_search_results_path <path> --dataset_path <path>

    Run 'COMMAND --help' for details on one command.
    """

    @staticmethod
    @cli_safe
    def index(max_chunk_size: PositiveInt = 2000) -> None:
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

        index_chunks(chunks, INDEX_DIR)
        print(f"Ingestion complete! Indices saved under {INDEX_DIR}")

    @staticmethod
    @cli_safe
    def search(query: str, k: PositiveInt = 5) -> None:
        """Print the top-k source locations of a single query.

        Args:
            query: Question to search for.
            k: Number of sources to return.
        """
        if not query.strip():
            print("Error: the query is empty.")
            return

        results = get_search([UnansweredQuestion(question=query)], k)
        if results:
            print_search_result(results)

    @staticmethod
    @cli_safe
    def search_dataset(
        dataset_path: str,
        save_directory: str,
        k: PositiveInt = 5
    ) -> None:
        """Search a whole dataset and save the StudentSearchResults JSON.

        Args:
            dataset_path: JSON file holding the questions.
            save_directory: Directory where the result file is written.
            k: Number of sources to return per question.
        """
        dataset = load_json_dataset(dataset_path)
        if dataset is None:
            return

        results = get_search(dataset.rag_questions, k)
        if results is None:
            return

        save_json(
            results,
            save_directory,
            Path(dataset_path).name,
            "student_search_results"
        )

    @staticmethod
    @cli_safe
    def answer(query: str, k: PositiveInt = 5) -> None:
        """Print the answer to a single query.

        Args:
            query: Question to answer.
            k: Number of sources used as context.
        """
        if not query.strip():
            print("Error: the query is empty.")
            return

        records = retrieve([query], k)
        if records is None:
            return

        print(generate_answer(query, records[0]))

    @staticmethod
    @cli_safe
    def answer_dataset(
            student_search_results_path: str, save_directory: str
    ) -> None:
        """Answer every question of a search-results file and save the JSON.

        Args:
            student_search_results_path: JSON file made by search_dataset.
            save_directory: Directory where the result file is written.
        """
        search_results = load_search_results(
            student_search_results_path)
        if search_results is None:
            return

        retriever = load_retriever()
        if retriever is None:
            return

        chunk_by_span = {
            (r["file_path"], r["first_character_index"],
             r["last_character_index"]): r
            for r in retriever.corpus}

        questions = search_results.search_results
        print(f"Loaded {len(questions)} questions")
        print("Loading model...")
        get_llm()

        answered = []
        for result in tqdm(questions, desc="Answering", unit="question"):
            spans = [(s.file_path, s.first_character_index,
                      s.last_character_index)
                     for s in result.retrieved_sources]
            chunks = [chunk_by_span[s] for s in spans if s in chunk_by_span]
            answered.append(MinimalAnswer(
                **result.model_dump(),
                answer=generate_answer(result.question, chunks)))

        output = StudentSearchResultsAndAnswer(
            k=search_results.k, search_results=answered)
        save_json(
            output, save_directory,
            Path(student_search_results_path).name,
            "student_search_results_and_answer"
        )

    @staticmethod
    @cli_safe
    def evaluate(
        student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print the recall@k of search results against a ground truth.

        Args:
            student_search_results_path: JSON file made by search_dataset.
            dataset_path: JSON file of answered questions (ground truth).
        """
        student = load_search_results(student_search_results_path)
        dataset = load_json_dataset(dataset_path)
        if student is None or dataset is None:
            return

        truth = [q for q in dataset.rag_questions
                 if isinstance(q, AnsweredQuestion) and q.sources]
        if not truth:
            print("Error: the dataset has no answered questions.")
            return

        retrieved = {r.question_id: r.retrieved_sources
                     for r in student.search_results}

        too_long = sum(
            (s.last_character_index - s.first_character_index)
            > MAX_SOURCE_LENGTH
            for sources in retrieved.values() for s in sources
        )

        missing = sum(q.question_id not in retrieved for q in truth)

        if too_long:
            print(f"Warning: {too_long} sources are longer than "
                  f"{MAX_SOURCE_LENGTH} characters.")

        if missing:
            print(f"Warning: {missing} questions have no search results.")

        ks = sorted({k for k in RECALL_KS if k <= student.k} | {student.k})
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        for q in truth:
            sources = retrieved.get(q.question_id, [])
            for k in ks:
                recalls[k].append(recall(q.sources, sources[:k]))

        print("Evaluation Results")
        print("=" * 40)
        print(" ".join(f"Recall@{k}: {sum(r) / len(r):.3f}"
                       for k, r in recalls.items()))


if __name__ == "__main__":
    fire.Fire(Rag, name="rag")
