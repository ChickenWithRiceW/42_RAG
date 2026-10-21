*This project has been created as part of the 42 curriculum by ascheufe.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a
codebase, here the **vLLM 0.10.1** repository. The pipeline:

1. **indexes** the repository (chunking + BM25 index saved on disk),
2. **retrieves** the most relevant snippets for a question, each returned as
   a source location (`file_path` + character range),
3. **generates** a grounded answer from those snippets with a small local
   model, `Qwen/Qwen3-0.6B`,
4. **measures** retrieval quality with recall@k.

Everything runs locally on CPU. Retrieval is purely lexical (BM25), tuned for
a corpus that mixes prose (docs) and code.

## Instructions

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/). The deep
learning stack (torch, transformers) and the model weights take several GB;
the model is downloaded from Hugging Face the first time `answer` or
`answer_dataset` runs.

```bash
make install          # uv sync
make run ARGS='search "How do I load a LoRA adapter?" --k 5'
make debug ARGS='search "lora"'   # same, under pdb
make lint             # flake8 . + mypy with the mandatory flags
make lint-strict      # flake8 . + mypy --strict
make clean            # remove caches
```

Expected layout (the corpus and datasets are provided, not committed):

```
data/raw/vllm-0.10.1/                       # the indexed repository
data/processed/                             # index, written by `index`
data/datasets/{Unanswered,Answered}Questions/
data/output/search_results/<DatasetScope>/              # search_dataset
data/output/search_results_and_answer/<DatasetScope>/   # answer_dataset
```

### Commands

All commands are run as `uv run python -m src <command> [options]`.

| Command | Purpose |
|---|---|
| `index --max_chunk_size 2000` | Ingest `data/raw/`, build the index under `data/processed/` |
| `search <query> --k 5` | Print the top-k sources of one query |
| `search_dataset --dataset_path <path> --k 10 --save_directory <Path>` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer <query> --k 5` | Answer one query from the retrieved context |
| `answer_dataset --student_search_results_path <Path> --save_directory <Path>` | Write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <Path> --dataset_path <Path>` | Print recall@k against a ground truth |

### Example usage

```bash
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indices saved under data/processed/

uv run python -m src search "What HTTP endpoint loads a LoRA adapter?" --k 3
# data/raw/vllm-0.10.1/docs/features/lora.md [4695:6098]
# ...

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# Recall@1: 0.630 Recall@3: 0.800 Recall@5: 0.860 Recall@10: 0.890

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
# This will take 100 bilion years I would not recommend running this on school hardware
```

Degenerate inputs never produce a traceback: empty query, `--k 0` or
non-numeric `--k` (exit code 2 with a message), missing dataset / results files,
missing index, unwritable output directory.

## System architecture

```
 data/raw/ ──► loader ──► chunks ──► tokenizer ──► BM25 index ──► data/processed/
 (py, md, txt)  (2 splitters)  (+ char spans)  (shared)            (bm25s + corpus.jsonl)

 question ──► tokenizer ──► BM25 search ──► top-k sources ──► JSON (search results)
                                                   │
                                                   ▼
                          context builder ──► Qwen3-0.6B ──► JSON (answers)
```

| Module | Role |
|---|---|
| `src/loader` | Walks the corpus, reads files, splits them into chunks with exact character spans |
| `src/helper` | Index build/load, retrieval, answer generation, recall, JSON I/O |
| `src/data_models` | Pydantic models exchanged between stages (those of the subject) |
| `src/type_checker` | `cli_safe` decorator: argument validation and clean error exits |
| `src/__main__.py` | Python Fire CLI (`Rag` class, one static method per command) |

## Chunking strategy

Two distinct strategies, both based on LangChain's
`RecursiveCharacterTextSplitter`, which cuts at the coarsest separator that
keeps a chunk under the size limit:

- **Python files**: language-aware separators (`class `, `def `, blank
  lines, newlines...), so functions and classes tend to stay whole.
- **Markdown / text files**: Markdown separators (headings, code fences,
  paragraphs), so sections stay together.

Other details:

- `--max_chunk_size` default **2000** and a 10 % overlap.
- Each chunk keeps its exact span: `text[first:last] == chunk`, so returned ranges point at the right place in the file.
- Indexed: `.py`, `.md`, `.txt`, `Makefile`, `Dockerfile`.
  `__pycache__`, `node_modules`, virtualenvs are skipped.
- `file_path` is written exactly as walked from `data/raw/`
  (`data/raw/vllm-0.10.1/...`).

## Retrieval method

**BM25** Ranking: chunks are scored against the query terms and the top-k
are returned best first; chunks with a score of 0 (no word in common with
the query) are dropped, so a nonsensical query returns no source rather than
arbitrary ones.

What is indexed matters more than the scoring function. The tokenizer:

1. keeps every identifier **whole** (`max_num_batched_tokens`) **and split**
   into sub-words (`max`, `num`, `batched`, `tokens`; camelCase too), so a
   question can quote an identifier *or* paraphrase it;
2. lower-cases, drops one-character tokens and a small English stopword
   list;
3. is applied to the **relative file path + chunk text**, so a question
   mentioning `fused_moe` or `lora` matches files named that way even when
   the chunk itself does not repeat the word.

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions),
one source per question, IoU threshold 0.05, `max_chunk_size=2000`:

| | R@1 | R@3 | R@5 | R@10 | Required R@5 |
|---|---|---|---|---|---|
| docs | 0.630 | 0.800 | **0.860** | 0.890 | 0.80 |
| code | 0.354 | 0.525 | **0.586** | 0.657 | 0.50 |


Effect of the chunk size (full pipeline, `index` then `search_dataset`):

| max_chunk_size | index time | docs R@5 | code R@5 |
|---|---|---|---|---|
| 500 | ~2 s | 0.790 | 0.596 |
| 1000 | ~1.7 s | 0.850 | 0.576 |
| 1500 | ~1.6 s | 0.840 | 0.596 |
| 2000 | ~1.5 s | 0.800 | 0.586 |


## Design decisions

- **BM25 only.** Simple, fast, no model at retrieval time, and strong on
  identifier-heavy questions once the tokenizer is right.
- **Path in the index, clean text in the corpus.** The path is tokenized
  but the stored chunk text is the raw file slice, so the LLM sees no
  artificial header.
- **Answer generation:** chunks are put in the prompt best first, each with
  its path and span, within a 10 000-character budget (about 3 000 tokens,
  far under Qwen3-0.6B's context); thinking mode is disabled and decoding
  is greedy; the prompt asks to copy exact names and to say so when the
  context lacks the answer. The model is loaded lazily, so retrieval
  commands start instantly.
- **Robust CLI:** every command is wrapped by `cli_safe` (typed argument
  validation with Pydantic, one-line errors, no tracebacks).
## Challenges faced

- **Chunks that do not name their topic.** A code chunk deep in a file often
  lacks the module's name; adding the file path to the indexed text made it better.
- **Exact spans.** The grader compares file paths verbatim and character
  ranges by overlap, so offsets were verified for every chunk and the path
  is never rewritten.
- **CLI with Python Fire.** Commands defined in a class need
  `@staticmethod`; without it Fire passes the class instance as the first
  argument. Fire also parses `123` as an int, so string arguments are
  coerced back to `str`.
- **Weak small model.** Qwen3-0.6B can only be helped by good retrieval and
  a strict prompt, hence the effort on recall.

## Resources

- `bm25s` documentation: https://github.com/xhluca/bm25s
- LangChain text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Python Fire: https://github.com/google/python-fire
- Pydantic: https://docs.pydantic.dev
- uv: https://docs.astral.sh/uv/

### Use of AI

An AI assistant (Claude) was used to: generate the readme. It was after rewritten and reviewed by me (Hello :D).