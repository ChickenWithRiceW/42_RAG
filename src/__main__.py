import fire
from src.loader import load_and_split_files
from src.type_checker import cli_safe, NoneNegativeInt, PositiveInt
import bm25s


INDEX_DIR = "data/processed"
RAW_DIR = "vllm-0.10.1"


def index_chunks(chunks, index_dir):
    records = [{"text": c.page_content,
                "file_path": c.metadata["source"],
                "first_character_index": c.metadata["start_index"],
                "last_character_index": c.metadata["end_index"]} for c in chunks]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([r["text"] for r in records], stopwords="en"))
    retriever.save(index_dir, corpus=records)

def search(query, k, index_dir):
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    docs, scores = retriever.retrieve(bm25s.tokenize([query], stopwords="en"), k=k)
    return docs[0]


class Rag:
    def other(self):
        print("other")

    @cli_safe
    def index(self, max_chunk_size: PositiveInt = 2000, raw_data_path: str = RAW_DIR,
                index_dir: str = INDEX_DIR, chunk_overlap: NoneNegativeInt = 200) -> None:
        """--max-chunk-size <max chunk size> --raw-data <raw data path>
        --index-dir <path to index> --chunk-overlap <size of overlap>
        """
        print("Hey!", max_chunk_size)
        d = load_and_split_files(raw_data_path, max_chunk_size, chunk_overlap)


        index_chunks(d, INDEX_DIR)
        txt = search("What license does vllm run on?", 5, INDEX_DIR)
        print(txt)

if __name__ == "__main__":
    fire.Fire(Rag, name="rag")


