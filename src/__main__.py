import fire

# from langchain_community.document_loaders import 

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter as RCTS


from src.type_checker import cli_safe, NoneNegativeInt, PositiveInt


INDEX_DIR = "data/processed"
RAW_DIR = "data/raw"




# ----- Type hints for CLI -----





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
        d = self._load_and_split_files(raw_data_path, max_chunk_size, chunk_overlap)
        print(d)


if __name__ == "__main__":
    fire.Fire(Rag, name="rag")


