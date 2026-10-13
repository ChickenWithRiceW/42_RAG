from langchain_community.document_loaders import PythonLoader, TextLoader, DirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document


class DataLoader:
    def __init__(self, chunk_size: int):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
        )
        self._code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=100,
        )

    def _load(self, path: str, glob: str):
        loader = DirectoryLoader(
            path,
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            recursive=True,
            silent_errors=True,  # skip files that fail to decode
        )
        return loader.load()

    def load(self, path: str):
        code_docs = self._load(path, "**/*.py")
        text_docs = self._load(path, "**/*.md") + self._load(path, "**/*.txt")

        code_chunks = self._code_splitter.split_documents(code_docs)
        text_chunks = self._text_splitter.split_documents(text_docs)

        for c in code_chunks:
            c.metadata["source_type"] = "code"
        for c in text_chunks:
            c.metadata["source_type"] = "text"

        return code_chunks + text_chunks


loader = DataLoader(chunk_size=1000)
chunks = loader.load("test/")
print(len(chunks))
print(chunks[0].metadata)
print(chunks[1].metadata)
print(chunks[2].metadata)

