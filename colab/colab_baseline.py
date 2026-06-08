import subprocess, sys, os, glob, shutil, datetime, time, json

REPO   = "https://github.com/almaas-izdihar/4167a324fe"
BRANCH = "experiment/confidence-filter"
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

def gpu_info():
    r = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True
    )
    if r.returncode != 0:
        return {}
    parts = [x.strip() for x in r.stdout.strip().split(',')]
    return {
        "gpu":          parts[0],
        "mem_total_mb": int(parts[1]),
        "mem_used_mb":  int(parts[2]),
        "util_pct":     int(parts[3]),
        "temp_c":       int(parts[4]),
    }

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
gpu_before = gpu_info()
t0 = time.time()

run(
    f"CUDA_VISIBLE_DEVICES=0 python3 main.py "
    f"--data_type cifar100 --data_path {DATA} "
    f"--classifier_type ResNet18 "
    f"--batch_size {BATCH_SIZE} --end_epoch {END_EPOCH} --workers {WORKERS} "
    f"--seed 2024 "
    f"--experiment_type s0_baseline"
)

duration = time.time() - t0
gpu_after = gpu_info()
os.makedirs("results", exist_ok=True)
report = {
    "model":          "baseline",
    "epochs":         END_EPOCH,
    "duration_sec":   round(duration, 1),
    "duration_human": f"{int(duration//60)}m{int(duration%60)}s",
    "gpu_before":     gpu_before,
    "gpu_after":      gpu_after,
}
with open("results/baseline_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"[report] duration={report['duration_human']}  gpu={gpu_before.get('gpu')}  "
      f"mem_before={gpu_before.get('mem_used_mb')}MB  mem_after={gpu_after.get('mem_used_mb')}MB",
      flush=True)

print("[colab_baseline] training done", flush=True)

# Push log to GitHub
GH_TOKEN = os.environ.get("GH_TOKEN", "")
if not GH_TOKEN:
    print("[push] GH_TOKEN not set — skipping push", flush=True)
else:
    logs = sorted(glob.glob("models/*EHSKD_False*/log/log.txt"))
    if not logs:
        print("[push] no log found — skipping", flush=True)
    else:
        os.makedirs("results", exist_ok=True)
        shutil.copy(logs[-1], "results/baseline_log.txt")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        remote = f"https://oauth2:{GH_TOKEN}@github.com/almaas-izdihar/4167a324fe"
        run("git config user.email 'almaasizdihar@gmail.com'")
        run("git config user.name 'almaas-izdihar'")
        run("git add results/baseline_log.txt")
        run(f"git commit -m 'results: baseline {ts}'")
        run(f"git push {remote} HEAD:{BRANCH}")
        print("[push] baseline_log.txt pushed", flush=True)

print("[colab_baseline] done", flush=True)
