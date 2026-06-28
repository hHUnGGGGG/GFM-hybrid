# GFM-RAG Development

# Requirements

| Name        | Installation                                                 | Purpose                                                                             |
| ----------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Python 3.12 | [Download](https://www.python.org/downloads/)                | The library is Python-based.                                                        |
| Poetry      | [Instructions](https://python-poetry.org/docs/#installation) | Poetry is used for package management and virtualenv management in Python codebases |

# Getting Started

## Install Dependencies
```shell
# install python dependencies
poetry install
```

## Install Pre-commit Hooks
Set up pre-commit hooks for development:

```bash
pre-commit install
```

## CUDA Installation
GFM-RAG require the `nvcc` compiler to compile the `rspmm` kernel. If you encounter errors related to CUDA, make sure you have the CUDA toolkit installed and the `nvcc` compiler is in your PATH. Meanwhile, make sure your CUDA_HOME variable is set properly to avoid potential compilation errors, e.g.,

```bash
export CUDA_HOME=/usr/local/cuda-12.4
```


## Repository Structure
An overview of the repository's top-level folder structure is provided below, detailing the overall design and purpose.

```shell
gfm_rag/                     # Root directory
├── docs/                    # Documentation
|   ├── DEVELOPING.md         # Development guide
|   |── CHANGELOG.md             # Project changelog
│   ├── config/             # Configuration documentation
│   │   ├── kg_index_config.md
│   │   └── ...
│   └── workflow/           # Workflow documentation
│       ├── kg_index.md
│       ├── training.md
│       └── ...
├── gfmrag/                 # Main package
|   ├── gfmrag_retriever.py # GFM-RAG retriever
|   ├── kg_indexer.py       # KG-index builder
|   ├── models.py          # GFM models
|   ├── losses.py       # Training losses
|   ├── doc_rankers.py   # Document rankers
│   ├── datasets/           # Dataset implementations
│   │   ├── qa_dataset.py
│   │   └── ...
│   ├── kg_construction/    # Knowledge graph construction
│   │   ├── entity_linking_model/ # Entity linking models
│   │   ├── ner_model/ # Named entity recognition models
│   │   ├── openie_model/ # OpenIE models
│   │   ├── kg_constructor.py # KG constructor
│   │   ├── qa_constructor.py # QA constructor
│   │   └── utils.py
│   ├── ultra/             # ultra models
│   │   ├── models.py
│   │   ├── layers.py
│   │   └── ...
|   ├── workflow/              # Training and inference scripts
|   │   ├── config/           # Configuration files
|   │   │   ├── stage1_index_dataset.yaml
|   │   │   ├── stage2_qa_finetune.yaml
|   │   │   ├── stage3_qa_inference.yaml
|   │   │   └── ...
|   │   ├── stage1_index_dataset.py
|   │   ├── stage2_qa_finetune.py
|   │   └── stage3_qa_inference.py
│   ├── llms/              # Language models
│   ├── evaluation/         # Evaluator for QA
│   └── utils/             # Utility functions
├── tests/                  # Test cases
├── scripts/                  # Scripts for running experiments
├── mkdocs.yml           # Documentation configuration
├── poetry.lock         # Poetry lock file
└── pyproject.toml      # Project configuration
```

## Common Commands

Serve the documentation locally:

```shell
mkdocs serve
```

Run the pre-commit hooks:

```shell
pre-commit run --all-files
```

Build package:

```shell
poetry build
```
