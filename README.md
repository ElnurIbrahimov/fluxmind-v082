# FluxMind v0.82 — Scaled Meta-Learning Experiment

Scaling experiment for FluxMind meta-learning. Run on RunPod GPU.

## What Changed (v0.81 → v0.82)

| | v0.81 | v0.82 |
|---|---|---|
| Params | 877K | 1.74M (2x) |
| Train DSLs | 80 | 161 (2x) |
| Test DSLs | 20 (imbalanced) | 42 (6/family, balanced) |
| Support set | 32 examples | 64 examples |
| Epochs | 3000 | 5000 |
| Scheduler | Cosine | Cosine warm restarts |

Target: 85%+ accuracy on novel DSLs (up from 78.6%).

## Run

```bash
pip install torch numpy
python train_v082_scaled.py --device cuda --epochs 5000 --batches 300
```

Results save to `v082_results/`.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- GPU with 8GB+ VRAM (RTX 4090 recommended)
