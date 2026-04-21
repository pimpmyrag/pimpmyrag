#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python export_with_optimum.py \
  --checkpoint /Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tunning-21042026/checkpoint_best_multitask.pt \
  --output /Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tunning-21042026/best_model_multitask.onnx \
  --model-name microsoft/deberta-v3-base > /tmp/onnx_export.log 2>&1
echo "Exit: $?"

