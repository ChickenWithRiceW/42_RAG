import fire
from src.loader import load_and_split_files
from src.type_checker import cli_safe, NoneNegativeInt, PositiveInt
from src.data_models import MinimalSource, StudentSearchResults, MinimalSearchResults
from pydantic import ValidationError

from langchain_huggingface import HuggingFacePipeline

import bm25s
import json


INDEX_DIR = "data/processed/"
RAW_DIR = "vllm-0.10.1"


class Rag:
    def other(self):
        print("other")

    @cli_safe
    def index(
        self,
        max_chunk_size: PositiveInt = 2000,
        raw_data_path: str = RAW_DIR,
        index_dir: str = INDEX_DIR,
        chunk_overlap: NoneNegativeInt = 200
    ) -> None:

        """--max_chunk_size <max chunk size> --raw_data <raw data path>
        --index_dir <path to index> --chunk_overlap <size of overlap>
        """
        chunks = load_and_split_files(
            raw_data_path, max_chunk_size, chunk_overlap)

        if not chunks:
            print("Error: Either path is empty or permission problem. "
                  "Could not load any files.")
            return
        self._index_chunks(chunks, index_dir)
        print(f"Ingestion complete! Indices saved under {index_dir}")

    def _index_chunks(self, chunks: list, index_dir: str):
        records = [{"text": c.page_content,
                    "file_path": c.metadata["source"],
                    "first_character_index": c.metadata["start_index"],
                    "last_character_index": c.metadata["end_index"]} for c in chunks]
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize([r["text"] for r in records], stopwords="en"))
        retriever.save(index_dir, corpus=records)

    @cli_safe
    def search(self, query: str, k: PositiveInt = 5,
               index_dir: str = INDEX_DIR) -> list[MinimalSource]:
        if not (anwser := self._get_result(query, k, index_dir)):
            return None

        self._print_search_result(anwser)
        self._export_search_result([anwser])

    @staticmethod
    def _get_result(query: str, k: PositiveInt, index_dir: str) -> None | StudentSearchResults:
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        docs, _ = retriever.retrieve(
            bm25s.tokenize([query], stopwords="en"), k=k)
        try:
            source_list = [MinimalSource(**d) for d in docs[0]]

            min_search = MinimalSearchResults(
                question=query,
                retrieved_sources=source_list,
            )

            stud_search = StudentSearchResults(
                k=k,
                search_results=[min_search],
            )
        except ValidationError as e:
            print(e)
            return None
        return stud_search

    @staticmethod
    def _get_context(query: str, k: PositiveInt, index_dir: str):
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        docs, _ = retriever.retrieve(
            bm25s.tokenize([query], stopwords="en"), k=k)
        # print(docs[0])
        return docs[0]
        # for d in docs[0]:
            # print(d)

    @staticmethod
    def _print_search_result(stud_search: StudentSearchResults) -> None:
        for result in stud_search.search_results[0].retrieved_sources:
            print(f"{result.file_path} [{result.first_character_index}:{result.last_character_index}]")

    @staticmethod
    def _export_search_result(stud_search: list[StudentSearchResults]) -> None:
        with open("sample.json", "w") as f:
            json.dump(json.loads(stud_search[0].model_dump_json(indent=4)), f, indent=4)

    @classmethod
    @cli_safe
    def answer(cls, query: str, k: PositiveInt = 5, index_dir: str = INDEX_DIR) -> None:
        docs = cls._get_context(query, k, index_dir)
        context = "\n\n".join(f"file_path: '{d["file_path"]}' text: '{d["text"]}'" for d in docs)        # chunk text, not the dicts
        prompt = (
            "Answer the question using only the context. Be specific: copy exact "
            "names, flags, commands and values from the context. If the context "
            f"does not contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {query}")
        llm = HuggingFacePipeline.from_model_id(
            model_id="Qwen/Qwen3-0.6B", task="text-generation",
            pipeline_kwargs={"max_new_tokens": 300, "do_sample": False,
                            "return_full_text": False})
        chat = llm.pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        print(llm.invoke(chat).strip())       # llm.invoke returns a plain str: no .co / .content
        print([d["file_path"] for d in docs], sep="\n")
        
    @classmethod
    @cli_safe
    def search_dataset(dataset_path: str, k: PositiveInt = 5, save_directory: str):
        


if __name__ == "__main__":
    fire.Fire(Rag, name="rag")
