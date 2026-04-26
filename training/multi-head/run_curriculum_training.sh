#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Training multi-phase avec curriculum de hard negatives
#  v2 — dataset augmenté (~3x) avec annotations Wikinews
# ═══════════════════════════════════════════════════════════
#
#  Phase 1 (epochs 1-2)  : négatifs faciles seulement, warm-up
#  Phase 2 (epochs 3-5)  : introduction progressive des englobants
#  Phase 3 (epochs 6-8)  : full hard negs, focal loss
#  Phase 4 (epochs 9-10) : stabilisation, LR très bas
#
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Activer le venv
source venv/bin/activate

MODEL="microsoft/deberta-v3-base"
TOKENIZER="$PWD/deberta/tokenizer_export"
DATA="data"
BS=16
ACCUM=2

echo "═══════════════════════════════════════════"
echo "  PHASE 1 : BUILD DATASET (easy negs only)"
echo "═══════════════════════════════════════════"

python3 data/build_multitask_dataset.py \
    --input $DATA/train.jsonl \
    --output $DATA/train.phase1.multitask.jsonl \
    --model-name $MODEL \
    \
    --englobant-ratio 0.0 \
    --hard-neg-ratio 0.5 \
    --englobant-weight 1.0

python3 data/build_multitask_dataset.py \
    --input $DATA/val.jsonl \
    --output $DATA/val.multitask.jsonl \
    --model-name $MODEL \
    \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

python3 data/build_multitask_dataset.py \
    --input $DATA/test.jsonl \
    --output $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

echo "═══════════════════════════════════════════"
echo "  PHASE 1 : TRAIN (epochs 1-2, easy negs)"
echo "═══════════════════════════════════════════"

python3 train_multi_task.py \
    --train $DATA/train.phase1.multitask.jsonl \
    --val $DATA/val.multitask.jsonl \
    --test $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    \
    --epochs 2 \
    --batch-size $BS \
    --accum-steps $ACCUM \
    --lr 7e-6 \
    --head-lr-multiplier 5.0 \
    --warmup-epochs 1 \
    --lambda-boundary 1.0 \
    --lambda-coarse 1.0 \
    --lambda-fine 0.5 \
    --focal-gamma 0.0 \
    --device cpu \
    2>&1 | tee logs/phase1.log

echo "═══════════════════════════════════════════"
echo "  PHASE 2 : BUILD DATASET (50% englobants)"
echo "═══════════════════════════════════════════"

python3 data/build_multitask_dataset.py \
    --input $DATA/train.jsonl \
    --output $DATA/train.phase2.multitask.jsonl \
    --model-name $MODEL \
    \
    --englobant-ratio 0.5 \
    --hard-neg-ratio 1.0 \
    --englobant-weight 1.0

echo "═══════════════════════════════════════════"
echo "  PHASE 2 : TRAIN (epochs 3-5, +englobants)"
echo "═══════════════════════════════════════════"

python3 train_multi_task.py \
    --train $DATA/train.phase2.multitask.jsonl \
    --val $DATA/val.multitask.jsonl \
    --test $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    \
    --epochs 5 \
    --start-epoch 3 \
    --batch-size $BS \
    --accum-steps $ACCUM \
    --lr 5e-6 \
    --head-lr-multiplier 4.0 \
    --warmup-epochs 0 \
    --lambda-boundary 1.5 \
    --lambda-coarse 1.0 \
    --lambda-fine 1.0 \
    --focal-gamma 1.0 \
    --device cpu \
    --resume checkpoint_last_multitask.pt \
    2>&1 | tee logs/phase2.log

echo "═══════════════════════════════════════════"
echo "  PHASE 3 : BUILD DATASET (75% englobants)"
echo "═══════════════════════════════════════════"

python3 data/build_multitask_dataset.py \
    --input $DATA/train.jsonl \
    --output $DATA/train.phase3.multitask.jsonl \
    --model-name $MODEL \
    \
    --englobant-ratio 0.75 \
    --hard-neg-ratio 1.0 \
    --englobant-weight 1.5

echo "═══════════════════════════════════════════"
echo "  PHASE 3 : TRAIN (epochs 6-8, 75% hard)"
echo "═══════════════════════════════════════════"

python3 train_multi_task.py \
    --train $DATA/train.phase3.multitask.jsonl \
    --val $DATA/val.multitask.jsonl \
    --test $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    \
    --epochs 8 \
    --start-epoch 6 \
    --batch-size $BS \
    --accum-steps $ACCUM \
    --lr 5e-6 \
    --head-lr-multiplier 3.0 \
    --warmup-epochs 0 \
    --lambda-boundary 2.0 \
    --lambda-coarse 1.0 \
    --lambda-fine 1.2 \
    --focal-gamma 1.5 \
    --device cpu \
    --resume checkpoint_best_multitask.pt \
    2>&1 | tee logs/phase3.log

echo "═══════════════════════════════════════════"
echo "  PHASE 4 : STABILISATION (epochs 9-10)"
echo "═══════════════════════════════════════════"

python3 train_multi_task.py \
    --train $DATA/train.phase3.multitask.jsonl \
    --val $DATA/val.multitask.jsonl \
    --test $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    \
    --epochs 10 \
    --start-epoch 9 \
    --batch-size $BS \
    --accum-steps $ACCUM \
    --lr 3e-7 \
    --head-lr-multiplier 4.0 \
    --warmup-epochs 0 \
    --lambda-boundary 2.0 \
    --lambda-coarse 1.0 \
    --lambda-fine 1.2 \
    --focal-gamma 1.5 \
    --device cpu \
    --resume checkpoint_best_multitask.pt \
    2>&1 | tee logs/phase4.log

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ TRAINING TERMINÉ"
echo "═══════════════════════════════════════════"

