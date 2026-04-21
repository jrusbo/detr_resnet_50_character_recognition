# detr_resnet_50_character_recognition

Digit detection project using Deformable DETR with a ResNet-50 backbone.

This repository is used to train, resume, evaluate, and run inference on a COCO-style
digit dataset (classes 1-10 in annotations, mapped to 0-9 internally).

## Project Structure

- `src/train.py`: main training entrypoint, checkpointing, resume logic, metric plots.
- `src/predict.py`: inference script that writes COCO-format predictions (`pred.json`).
- `src/dataset.py`: COCO dataset loaders, annotation sanitization, batch collator.
- `src/models.py`: model definition (`DetrResnet50`).
- `datasets/`: data folders and COCO annotations.
- `models/`: run outputs (checkpoints, best model, plots, metrics).

## Environment

- Python: `3.13+`
- Dependency manager: `uv`
- Lint style: Ruff (`E`/`F`, line length `100`)

Install dependencies:

```bash
uv sync
```

This project requires internet access during training/inference startup to download
`SenseTime/deformable-detr` assets from Hugging Face Hub.

## Training

Main command:

```bash
python src/train.py --data-root datasets
```

Common experiment example:

```bash
python src/train.py --epochs 20 --batch-size 16 --weight-decay 2e-4 --lr-backbone 5e-6
```

### What `train.py` saves

- Run directory: `models/run_YYYYMMDD_HHMMSS/`
- Periodic checkpoints: `checkpoint-<step>/`
- Best AP50 model: `best_map_model.pt`
- Best AP50 scalar: `best_map_value.txt`
- Plots: `loss_curves.png`, `confusion_matrix.png`, `pr_curves.png`

### Defaults (key arguments)

- `--epochs 50`
- `--batch-size 8`
- `--eval-batch-size 16`
- `--lr 2e-4`
- `--lr-backbone 2e-5`
- `--weight-decay 1e-4`
- `--warmup-ratio 0.05`
- `--max-grad-norm 0.1`
- `--min-size 200`
- `--max-size 400`
- `--num-workers 4`
- `--eval-every-epochs 2`
- `--map-eval-freq 5`
- `--metrics-threshold 0.3`
- `--early-stopping-patience 3` (`0` disables early stopping)
- `--early-stopping-threshold 0.0`
- `--max-train-hours 11.8`
- `--use-class-balanced-sampler` (off by default)

### Resume training

Resume from a run directory (auto-picks latest `checkpoint-*`):

```bash
python src/train.py --resume-from-checkpoint models/run_20260419_160432 --epochs 40
```

Resume from a specific checkpoint:

```bash
python src/train.py --resume-from-checkpoint models/run_20260419_160432/checkpoint-101500 --epochs 40
```

Important:

- `--epochs` is the total target epoch count, not an increment.
- Keep core settings consistent when resuming (`batch-size`, number of processes, data path).

## Inference

Generate `pred.json` from trained weights:

```bash
python src/predict.py --weights models/run_YYYYMMDD_HHMMSS/best_map_model.pt --data datasets/test
```

Optional arguments:

- `--output` (default: same folder as weights, file `pred.json`)
- `--batch-size` (default `8`)
- `--num-workers` (default `2`)
- `--threshold` (default `0.1`)
- `--min-size` / `--max-size` (default `200` / `400`)

## Metrics and Threshold Notes

- Training-time best-model selection uses AP50 with threshold `0.2` (`MAP_SCORE_THRESHOLD`).
- Final confusion matrix and PR plots use `--metrics-threshold` (default `0.3`).
- For COCO-style AP50:95 evaluation experiments, lower score thresholds (for candidate generation)
  are often better than aggressive filtering.

## Documentation and Code Style

- Source files include module headers and symbol docstrings.
- Non-obvious implementation details are documented with brief comments.
- Style follows PEP 8 and the repository Ruff configuration.
