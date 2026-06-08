import subprocess, sys, os, glob, shutil, datetime, time, json, threading, re

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

def run_training(cmd, total_epochs):
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _log_stop = threading.Event()

    def _log_watcher():
        log_path, log_pos = None, 0
        while not _log_stop.is_set():
            if log_path is None:
                found = glob.glob("models/*/log/log.txt")
                if found:
                    log_path = found[-1]
            if log_path:
                try:
                    with open(log_path) as f:
                        f.seek(log_pos)
                        for line in f:
                            m = re.search(r'\[val\] \[Epoch (\d+)\].*\[val_top1_acc ([\d.]+)\].*\[val_loss ([\d.]+)\]', line)
                            if m:
                                ep = int(m.group(1)) + 1
                                print(f">>> [{ep}/{total_epochs}] top1={m.group(2)} val_loss={m.group(3)}", flush=True)
                        log_pos = f.tell()
                except (IOError, OSError):
                    pass
            _log_stop.wait(2)

    watcher = threading.Thread(target=_log_watcher, daemon=True)
    watcher.start()
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    _log_stop.set()
    watcher.join(timeout=5)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

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

# GPU timeseries sampler
_samples = []
_stop = threading.Event()

def _sampler(interval=5):
    t0 = time.time()
    while not _stop.is_set():
        r = subprocess.run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
            "--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        if r.returncode == 0:
            try:
                p = [x.strip() for x in r.stdout.strip().split(',')]
                _samples.append({"t": round(time.time()-t0, 1),
                                  "util": int(p[0]), "mem_mb": int(p[1]), "temp_c": int(p[2])})
            except (ValueError, IndexError):
                pass
        _stop.wait(interval)

_t = threading.Thread(target=_sampler, args=(5,), daemon=True)
_t.start()

# Train
gpu_before = gpu_info()
t0 = time.time()

run_training(
    f"COLUMNS=80 CUDA_VISIBLE_DEVICES=0 python3 -u main.py "
    f"--data_type cifar100 --data_path {DATA} "
    f"--classifier_type ResNet18 "
    f"--batch_size {BATCH_SIZE} --end_epoch {END_EPOCH} --workers {WORKERS} "
    f"--seed 2024 "
    f"--experiment_type s0_baseline",
    total_epochs=END_EPOCH
)

duration = time.time() - t0
_stop.set(); _t.join(timeout=10)
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
with open("results/baseline_gpu_timeseries.json", "w") as f:
    json.dump(_samples, f)
print(f"[report] duration={report['duration_human']}  gpu={gpu_before.get('gpu')}  "
      f"samples={len(_samples)}", flush=True)

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
