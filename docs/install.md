# Installation

## Requirements

| Component | Requirement |
|---|---|
| Python | **3.12** (>=3.12, <3.13) |
| GPU | NVIDIA + **CUDA 12.x** (required for GFM/GNN) |
| LLM API | OpenAI key (OpenAI-compatible endpoint, e.g. Yescale) |

## Setup

```bash
conda create -n gfmhybrid python=3.12 && conda activate gfmhybrid
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .            # install the gfmrag_hybrid package (editable)
```

Create `.env` in the project root (do NOT commit):

```dotenv
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```