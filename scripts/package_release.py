"""Assemble a Hugging Face release folder from a training run's selected checkpoint (nothing is uploaded).

    python scripts/package_release.py runs/omni3b-B5aug-all release/EmotionDecider-AV-3B

Writes decision.safetensors (adapters + head, flat keys "<group>.<name>"), decision.json, calibration.json, the model
card (release/README.md), NOTICE, and the Qwen license copied from the pinned backbone snapshot. Checks that the
packaged weights reload to exactly the run's tensors.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from emotion_decider.inference import flatten, unflatten  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="training run directory (uses best/ and calibration.json)")
    ap.add_argument("out")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(run / "best" / "decision.pt", map_location="cpu", weights_only=True)
    meta = json.loads((run / "best" / "decision.json").read_text())
    save_file(flatten(state), out / "decision.safetensors", metadata={"format": "pt", "step": str(meta.get("step"))})
    back = unflatten(load_file(out / "decision.safetensors"))
    assert back.keys() == state.keys() and all(
        back[g].keys() == state[g].keys() and all(torch.equal(back[g][k], state[g][k]) for k in state[g]) for g in state)
    (out / "decision.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(run / "calibration.json", out / "calibration.json")
    shutil.copy(REPO / "release" / "README.md", out / "README.md")
    shutil.copy(REPO / "NOTICE", out / "NOTICE")
    from huggingface_hub import hf_hub_download
    b = meta["backbone"]
    shutil.copy(hf_hub_download(b["name"], "LICENSE", revision=b["revision"]), out / "LICENSE")
    for f in sorted(out.iterdir()):
        print(f"{f.name:<24} {f.stat().st_size / 1e6:8.1f} MB  sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
