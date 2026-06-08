import subprocess, sys, os

REPO   = "https://github.com/almaas-izdihar/ema-skd"
BRANCH = "experiment/colab-conf-filter"
DIR    = "/content/ema-skd"
DATA   = "/content/data"

END_EPOCH  = int(os.environ.get("END_EPOCH",  "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
WORKERS    = int(os.environ.get("WORKERS",    "4"))

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
    f"--beta 0.5 --EHSKD "
    f"--confidence_gate --tau_max 0.7 --tau_min 0.1 "
    f"--experiment_type s1_emaskd_conf_gate"
)

print("[colab_emaskd] done", flush=True)
