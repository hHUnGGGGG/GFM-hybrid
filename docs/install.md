# Installation Guide

## Prerequisites

Before installing GFM-RAG, make sure your system meets these requirements:

- Python 3.12 or higher
- CUDA 12 or higher (for GPU support)
- Poetry (recommended for development)

## Installation Methods

### Install via Conda
Conda provides an easy way to install the CUDA development toolkit which is required by GFM-RAG:

```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
pip install gfmrag
```

### Install via Pip

```bash
pip install gfmrag
```


### Install from Source

For contributors or those who want to install from source, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/RManLuo/gfm-rag.git
cd gfm-rag
```

2. Install [Poetry](https://python-poetry.org/docs/):

3. Create and activate a conda environment:
```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
```

4. Install project dependencies:
```bash
poetry install
```

## Optional Components

### Llama.cpp Integration

If you plan to use locally host LLMs via Llama.cpp:

Install llama-cpp-python:
```bash
pip install llama-cpp-python
```

For more information, visit the following resources:
- [LangChain Llama.cpp](https://python.langchain.com/docs/integrations/chat/llamacpp/)
- [llama-cpp-python repository](https://github.com/abetlen/llama-cpp-python)

### Ollama Integration

If you plan to use Ollama for hosting LLMs:

Install Ollama:
```bash
pip install langchain-ollama
pip install ollama
```

For more information, visit the following resources:
- [LangChain Ollama](https://python.langchain.com/docs/integrations/chat/ollama/)

## Troubleshooting


### CUDA errors when compiling `rspmm` kernel
GFM-RAG requires the `nvcc` compiler to compile the `rspmm` kernel. If you encounter errors related to CUDA, make sure you have the CUDA toolkit installed and the `nvcc` compiler is in your PATH. Meanwhile, make sure your CUDA_HOME variable is set properly to avoid potential compilation errors, eg

```bash
export CUDA_HOME=/usr/local/cuda-12.4
```

Usually, if you install CUDA toolkit via conda, the CUDA_HOME variable is set automatically.

### Stuck when compiling `rspmm` kernel

Sometimes the compilation of the `rspmm` kernel may get stuck. If you encounter this issue, try to manually remove the compilation cache under `~/.cache/torch_extensions/` and recompile the kernel.

For more help, please check our [GitHub issues](https://github.com/RManLuo/gfm-rag/issues) or create a new one.
