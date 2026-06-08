import subprocess, sys, os

REPO   = "https://github.com/almaas-izdihar/4167a324fe"
BRANCH = "experiment/confidence-filter"
DIR    = "/content/ema-skd"
DATA   = "/content/data"

END_EPOCH  = 2
BATCH_SIZE = 128
WORKERS    = 4

def run(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        sys.exit(r.returncode)

# Clone
if not os.path.exists(DIR):
    run(f"git clone {REPO} {DIR}")
run(f"git -C {DIR} checkout {BRANCH}")
run(f"git -C {DIR} log --oneline -3")

os.chdir(DIR)

# Pre-download CIFAR-100
os.makedirs(DATA, exist_ok=True)
import torchvision
torchvision.datasets.CIFAR100(DATA, train=True,  download=True)
torchvision.datasets.CIFAR100(DATA, train=False, download=True)
print("CIFAR-100 ready.", flush=True)

# Train
run(
    f"CUDA_VISIBLE_DEVICES=0 python3 main.py "
    f"--data_type cifar100 --data_path {DATA} "
    f"--classifier_type ResNet18 "
    f"--batch_size {BATCH_SIZE} --end_epoch {END_EPOCH} --workers {WORKERS} "
    f"--seed 2024 "
    f"--experiment_type s0_baseline_smoke"
)

print("[colab_baseline_smoke] done", flush=True)
