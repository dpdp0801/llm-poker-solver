# LLM-GTO

This repository contains experiments for building a Llama‑3 based poker assistant. The project is split into multiple stages of fine‑tuning and evaluation.

## Setup

1. Install dependencies:

```bash
./setup.sh
```

2. Configure Hugging Face access:
   - Copy `.env.template` to `.env` and add your token.
   - The code loads this token with `python-dotenv`.

3. Run tests:

```bash
pytest -q
```

## Repository layout

- `data/` – raw training articles, books and transcripts.
- `scripts/` – entry points such as `train_stageA.py` and evaluation scripts.
- `src/llm_poker_solver/` – library code (e.g. `utils.py`).
- `solver/` – solver resources like preflop charts.
- `out/` – fine‑tuned model adapters (empty placeholder directories).

