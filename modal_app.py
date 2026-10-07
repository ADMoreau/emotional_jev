"""Run facedecide on Modal (structure follows Kev's modal_app.py, Apache-2.0).

    modal run modal_app.py::data                                   # CPU: CREMA-D -> crops, targets, manifests (once)
    modal run modal_app.py::smoke                                  # GPU: real Omni-3B, 2 train steps + 4-clip eval
    modal run modal_app.py::baseline --fold 0                      # GPU: zero-shot baseline on a test fold
    modal run --detach modal_app.py::train --fold 0 --name omni3b-B-f0
    modal run --detach modal_app.py::folds --name omni3b-B         # all 5 folds in parallel (5 GPUs)
    modal volume get facedecide-runs omni3b-B-f0 runs/             # results back to the laptop

Volumes (private to the workspace): facedecide-data (CREMA-D + derived crops/targets/manifests: licensed data, never
put in the image), facedecide-hf (Hugging Face cache: Omni-3B downloads once), facedecide-runs (checkpoints,
metrics, predictions). GPU type: FACEDECIDE_GPU (default A100-80GB). Training commits the runs Volume after every
checkpoint and retries on preemption, resuming from the last checkpoint.
"""
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GPU = os.environ.get("FACEDECIDE_GPU", "A100-80GB")
DATA, HF, RUNS = "/vol/data", "/vol/hf", "/vol/runs"
CREMA_D = "https://github.com/CheyneyComputerScience/CREMA-D.git"

app = modal.App("facedecide")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "git-lfs")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .pip_install("facenet-pytorch==2.6.0", extra_options="--no-deps")
    .env({"HF_HOME": HF, "PYTHONPATH": "/root/src", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/root")
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
    .add_local_dir(ROOT / "configs", "/root/configs")
    .add_local_dir(ROOT / "tests", "/root/tests")
)
data_vol = modal.Volume.from_name("facedecide-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("facedecide-hf", create_if_missing=True)
runs_vol = modal.Volume.from_name("facedecide-runs", create_if_missing=True)
GPU_VOLUMES = {DATA: data_vol, HF: hf_vol, RUNS: runs_vol}


def _link_data():
    """Scripts address data as ./data/crema-d; point that at the data Volume."""
    os.chdir("/root")
    if not Path("/root/data").exists():
        Path("/root/data").symlink_to(DATA)


def _sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


@app.function(image=image, cpu=16, memory=32768, timeout=4 * 3600, volumes={DATA: data_vol})
def prepare_data():
    """Idempotent: clone CREMA-D, pull media, verify checksums, then targets -> audio offsets -> face crops ->
    audit -> manifests. The splits and exclusion list it derives must equal the committed ones."""
    _link_data()
    repo = Path(DATA) / "crema-d"
    if not (repo / ".git").exists():
        _sh(f"GIT_LFS_SKIP_SMUDGE=1 git clone -q {CREMA_D} {repo}")
    _sh("git lfs install --local && git lfs pull --include 'VideoFlash/*,AudioWAV/*'", cwd=repo)
    _sh("git lfs ls-files -l -I 'VideoFlash/*,AudioWAV/*' | awk '{print $1\"  \"$3}' | sha256sum --quiet -c", cwd=repo)
    data_vol.commit()
    committed = {p.name: p.read_bytes() for p in Path("/root/configs/splits").glob("*.json")}
    env = "OMP_NUM_THREADS=1"
    _sh("python scripts/build_crema_targets.py")
    _sh(f"{env} python scripts/audio_offsets.py --workers 16")
    data_vol.commit()
    _sh(f"{env} python scripts/extract_faces.py --workers 16")
    data_vol.commit()
    _sh("python scripts/audit_crema.py")
    # The committed splits and exclusion list are authoritative, so every machine trains and tests on the same
    # clips. A rebuilt list can differ only through decoder numerics (e.g. 1080_TSI_DIS_XX matches its video at
    # corr 0.497, next to the 0.5 cutoff); report it, then restore the committed copy.
    for name, before in committed.items():
        path = Path("/root/configs/splits", name)
        if path.read_bytes() != before:
            import json
            new, old = json.loads(path.read_bytes()), json.loads(before)
            if "clips" in old and isinstance(old["clips"], dict):
                print(f"WARNING {name}: rebuilt list differs; only rebuilt: {sorted(set(new['clips']) - set(old['clips']))}, "
                      f"only committed: {sorted(set(old['clips']) - set(new['clips']))}. Using the committed list.")
            else:
                print(f"WARNING {name}: rebuilt copy differs; using the committed one.")
            path.write_bytes(before)
    _sh("python scripts/build_manifests.py --seed 0")
    data_vol.commit()
    pack_clips.local()


@app.function(image=image, cpu=16, memory=16384, timeout=2 * 3600, volumes={DATA: data_vol})
def pack_clips():
    """One file per clip (crops + WAV + meta) so a clip is one Volume read, not 18. Idempotent."""
    _link_data()
    _sh("python scripts/pack_clips.py --workers 16")
    data_vol.commit()


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, timeout=24 * 3600, volumes=GPU_VOLUMES,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train_fold(fold: int, name: str, overrides: list[str], max_steps: int = 0, limit_clips: int = 0,
               eval_limit: int = 0):
    _link_data()
    import torch
    from facedecide.train import load_config, train
    cfg = load_config("configs/omni_3b.yaml", overrides)
    torch.manual_seed(cfg["seed"])
    res = train(cfg, fold, f"{RUNS}/{name}", max_steps=max_steps or None, limit_clips=limit_clips or None,
                eval_limit=eval_limit or None, on_checkpoint=runs_vol.commit,
                log=lambda m: print(f"[{name}] {m}", flush=True))   # parallel runs share the app's log stream
    runs_vol.commit(); hf_vol.commit()
    return {k: v for k, v in res.items() if k != "dev"}


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, timeout=6 * 3600, volumes=GPU_VOLUMES)
def zero_shot(fold: int, name: str, calibrate: bool = True, limit_clips: int = 0):
    _link_data()
    cmd = f"python -m facedecide.zeroshot --config configs/omni_3b.yaml --fold {fold} --out {RUNS}/{name}"
    cmd += " --calibrate" * calibrate + (f" --limit_clips {limit_clips}" if limit_clips else "")
    _sh(cmd)
    runs_vol.commit(); hf_vol.commit()
    import json
    return json.loads(Path(RUNS, name, "metrics.json").read_text())


@app.local_entrypoint()
def data():
    prepare_data.remote()


@app.local_entrypoint()
def pack():
    pack_clips.remote()


@app.local_entrypoint()
def smoke(tag: str = "", clips: int = 16, steps: int = 6):
    """Real weights end to end, cheap (~10 min on an A100): `steps` optimizer steps of 2 clips each, calibration
    + scoring of `clips` dev and test clips (timings printed), and a `clips`-clip zero-shot pass. Use a new `tag`
    per run: training resumes from any checkpoint already in its output folder."""
    print(train_fold.remote(0, f"smoke-train{tag}", ["train.grad_accum=2", "train.log_every=1"],
                            max_steps=steps, limit_clips=clips))
    print(zero_shot.remote(0, f"smoke-zs{tag}", calibrate=True, limit_clips=clips))


@app.local_entrypoint()
def check(tag: str = "", steps: int = 200, eval_clips: int = 64):
    """Short real training run on fold 0 (~10-15 min on an A100): `steps` optimizer steps over the full training
    set (8 clips each), dev probes every 50 steps, then calibration and scoring on `eval_clips` dev/test clips.
    Shows whether the loss falls, the grad norm stays bounded and the GPU is kept busy."""
    print(train_fold.remote(0, f"check{tag}", [], max_steps=steps, eval_limit=eval_clips))


@app.local_entrypoint()
def baseline(fold: int = 0, name: str = "", calibrate: bool = True):
    print(zero_shot.remote(fold, name or f"zs-f{fold}", calibrate))


@app.local_entrypoint()
def train(fold: int = 0, name: str = "", set: str = "", max_steps: int = 0):
    print(train_fold.remote(fold, name or f"omni3b-f{fold}", set.split(), max_steps))


@app.local_entrypoint()
def folds(name: str = "omni3b", set: str = ""):
    """All 5 actor-disjoint folds in parallel, one GPU each."""
    for fold, res in enumerate(train_fold.starmap([(k, f"{name}-f{k}", set.split()) for k in range(5)])):
        print(fold, res.get("test", {}).get("intended", {}).get("av"))
