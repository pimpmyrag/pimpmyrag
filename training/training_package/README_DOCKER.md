# Containerized training (Docker)

This README explains how to build and run a Docker image for training the SpanClassifier and exporting the trained model to the host.

Prerequisites
- Docker installed and running on your machine.
- Enough disk space and network connectivity to download model weights.

Build the Docker image (run from repository root):

```bash
docker build -t pimpmyrag-trainer -f training/training_package/Dockerfile .
```

Run training inside the container and export the trained model to a host directory `training_output`:

```bash
# create output dir on host
mkdir -p training_output

# example run: adjust paths to your train/val/test files in the repo
docker run --rm -it \
  -v "$(pwd)":/app \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$(pwd)/training_output":/app/output \
  pimpmyrag-trainer -- --train /app/data/train.jsonl --val /app/data/val.jsonl --test /app/data/test.jsonl --epochs 6 --batch 16 --lr 2e-5
```

Notes
- We mount the repo into `/app` so the container uses your local code and data.
- The Hugging Face cache is mounted so downloaded models are reused between local runs and container runs.
- The container copies `best_model.pt` to `/app/output` after training — this will be available on the host under `training_output/best_model.pt`.
- If you need GPU support, you'll need a different base image and to configure `--gpus` when running the container (Linux + NVIDIA required).

Troubleshooting
- If the tokenizer fails parsing `spm.model`, the Docker run script clears the cached model directory to force a fresh download.
- If PyTorch wheel selection is an issue, adjust the `Dockerfile` to install the correct `torch` package for your platform.
