from src.loader import load_and_split_files
from src.type_checker import cli_safe, PositiveInt
from src.data_models import MinimalSource, StudentSearchResults, \
    MinimalSearchResults, RagDataset, UnansweredQuestion, AnsweredQuestion
from pydantic import ValidationError
from functools import cache
from pathlib import Path

from langchain_huggingface import HuggingFacePipeline

import fire
import bm25s
import json


INDEX_DIR = "data/processed/"
RAW_DIR = "data/"


class Rag:
    @cli_safe
    @classmethod
    def index(cls, max_chunk_size: PositiveInt = 2000) -> None:
        """--max_chunk_size <max chunk size>"""
        chunk_overlap = int(max_chunk_size * 0.1)
        chunks = load_and_split_files(RAW_DIR, max_chunk_size, chunk_overlap)

        if not chunks:
            print("Error: Either path is empty or permission problem. "
                  "Could not load any files.")
            return

        cls._index_chunks(chunks, INDEX_DIR)
        print(f"Ingestion complete! Indices saved under {INDEX_DIR}")

    def _index_chunks(self, chunks: list, index_dir: str):
        records = [
            {"text": c.page_content,
             "file_path": c.metadata["source"],
             "first_character_index": c.metadata["start_index"],
             "last_character_index": c.metadata["end_index"]
             } for c in chunks]

        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(
            [r["text"] for r in records],
            stopwords="en")
        )
        retriever.save(index_dir, corpus=records)

    @cli_safe
    @classmethod
    def search(cls, query: str, k: PositiveInt = 5) -> list[MinimalSource]:
        anwser = cls._get_search(
            [UnansweredQuestion(question=query)],
            k,
            INDEX_DIR
        )

        if not anwser:
            return None

        cls._print_search_result(anwser)

    @staticmethod
    def _get_search(questions: list[UnansweredQuestion], k: PositiveInt
                    ) -> None | StudentSearchResults:
        if not questions:
            print("Error: no queries given.")
            return None

        queries = [q.question for q in questions]
        question_ids = [q.question_id for q in questions]

        retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
        docs, scores = retriever.retrieve(
            bm25s.tokenize(queries, stopwords="en"),
            k=min(k, len(retriever.corpus)))

        ids = question_ids or [None] * len(queries)
        try:
            results = [
                MinimalSearchResults(
                    **({"question_id": qid} if qid else {}),
                    question=query,
                    retrieved_sources=[MinimalSource(**d)
                                    for d, s in zip(hits, sc) if s > 0])
                for query, qid, hits, sc in zip(queries, ids, docs, scores)]
            return StudentSearchResults(k=k, search_results=results)
        except ValidationError as e:
            print(e)
            return None

    @staticmethod
    def _get_context(query: str, k: PositiveInt, index_dir: str):
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        docs, _ = retriever.retrieve(
            bm25s.tokenize([query], stopwords="en"), k=k)
        return docs[0]

    @staticmethod
    def _print_search_result(stud_search: StudentSearchResults) -> None:
        for result in stud_search.search_results[0].retrieved_sources:
            print(f"{result.file_path} [{result.first_character_index}:{result.last_character_index}]")

    @staticmethod
    def _export_search_result(stud_search: StudentSearchResults, export_dir : str) -> None:
        path = Path(export_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_dir, "w") as f:
            json.dump(json.loads(stud_search.model_dump_json(indent=4)), f, indent=4)

    @cache
    def _llm():
        return HuggingFacePipeline.from_model_id(
            model_id="Qwen/Qwen3-0.6B", task="text-generation",
            pipeline_kwargs={"max_new_tokens": 300, "do_sample": False,
                            "return_full_text": False})

    @classmethod
    @cli_safe
    def answer(cls, query: str, k: PositiveInt = 5, index_dir: str = INDEX_DIR) -> None:
        docs = cls._get_context(query, k, index_dir)
        context = "\n\n".join(f"file_path: '{d["file_path"]}' text: '{d["text"]}'" for d in docs)        # chunk text, not the dicts

        prompt = (
            "Answer the question using only the context. Be specific: copy exact "
            "names, flags, commands and values from the context. If the context "
            f"does not contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {query}")
        llm = cls._llm()
        chat = llm.pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        print(llm.invoke(chat).strip())       # llm.invoke returns a plain str: no .co / .content
        print([d["file_path"] for d in docs], sep="\n")

    @classmethod
    @cli_safe
    def search_dataset(cls, dataset_path: str, save_directory: str, k: PositiveInt = 5):
        data_set = cls._load_json_dataset(dataset_path)
        
        res = cls._get_search(data_set.rag_questions, k, index_dir=INDEX_DIR)
        dataset_filename = Path(dataset_path).name
        save_dir = Path(save_directory)
        
        export_file = f"{save_dir.absolute()}/{dataset_filename}"
        # exit()
        cls._export_search_result(res, export_file)

    @staticmethod
    def _load_json_dataset(dataset_path: str) -> RagDataset | None:
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(e)
            return None

        res_list = []
        for d in data["rag_questions"]:
            if d.get("answer"):
                res_list.append(AnsweredQuestion(**d))
            else:
                res_list.append(UnansweredQuestion(**d))
        return RagDataset(rag_questions=res_list)


if __name__ == "__main__":
    fire.Fire(Rag, name="rag")
