#!/bin/bash
set -e

DATA=/content/data
EP=${END_EPOCH:-200}
BS=${BATCH_SIZE:-128}
WK=${WORKERS:-4}

echo "[run_baseline] epochs=$EP batch=$BS workers=$WK"

# Pre-download CIFAR-100
python3 -c "
import torchvision
torchvision.datasets.CIFAR100('$DATA', train=True,  download=True)
torchvision.datasets.CIFAR100('$DATA', train=False, download=True)
print('CIFAR-100 ready.')
"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --data_type cifar100 \
  --data_path $DATA \
  --classifier_type ResNet18 \
  --batch_size $BS \
  --end_epoch $EP \
  --workers $WK \
  --seed 2024 \
  --experiment_type s0_baseline

echo "[run_baseline] done"
