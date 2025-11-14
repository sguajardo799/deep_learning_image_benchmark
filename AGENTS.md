# Repository Guidelines
This guide summarizes how to contribute productively to Vision_DL_Benchmark.

## Project Structure & Module Organization
All runtime logic sits in `src/`: `data.py` manages dataset download/loading, `models.py` builds torchvision/timm backbones, `train.py` wraps the optimizer/early-stopping loop, and `utils.py` keeps helpers such as plotting. The CLI entrypoint is `main.py`, which routes modes defined in YAML. Default settings live in `config/default.yaml`; keep additional configs beside it with descriptive names. Artifacts (downloaded Hugging Face datasets, checkpoints, plots) land in `data/` or `output/` and can be regenerated; never commit them.

## Build, Test, and Development Commands
- `conda create -n image_dl python=3.10 && conda activate image_dl`: provision the canonical local environment before installing dependencies.
- `pip install -r requirements.txt`: installs PyTorch, datasets, torchvision, and plotting utilities inside `image_dl`.
- `python main.py --mode download_data --config config/default.yaml`: hydrates the dataset declared in the config into `data/`.
- `python main.py --mode train_model --config config/default.yaml`: launches the training routine (override dataset/model/task via CLI flags during experiments).
- `docker build -t vision-benchmark .` and `docker run --rm -v $(pwd)/data:/app/data vision-benchmark --mode train_model`: reproduce the reference container workflow documented in README.

## Coding Style & Naming Conventions
Follow PEP8/PEP257 with 4-space indentation and `snake_case` identifiers (see `src/models.py`). Prefer explicit names that echo config keys (`dataset_name`, `model_name`). Keep helper functions pure and side-effect free unless they touch logging or I/O.

## Testing Guidelines
Author tests under `tests/` using `pytest`, mirroring the module hierarchy (e.g., `tests/test_data.py`). Cover dataset downloads/transforms, CLI argument merging, and at least one abbreviated training/evaluation step on a toy subset to catch device/config issues. Run `pytest -q` before submitting; aim for coverage of every `mode` supported by `main.py`.

## Commit & Pull Request Guidelines
History currently uses short imperative messages (“Initial commit”); continue that tone and reference issue IDs when relevant. Pull requests should summarize behavior changes, note config deltas, attach training logs or curves when behavior shifts, and confirm `pytest` plus Docker builds ran cleanly. Keep scope tight (one benchmark or feature per PR) and link supporting experiments.

## Configuration & Data Handling
Do not mutate `config/default.yaml` while iterating; copy it to `config/<feature>.yaml` and point the CLI via `--config`. Document new YAML keys in both the file comments and PR. Large datasets, cache folders, and outputs belong in `data/` or `output/`; ensure they stay untracked and use Docker volume mounts or symlinks instead of copying them into the image.
