"""Run the 5-fold CREMA-D protocol on Modal end to end: deploy, launch, follow, collect, summarize.

    python scripts/run_5fold.py run --name omni3b-v2                    # folds 0-4 -> runs omni3b-v2-f0 .. -f4
    python scripts/run_5fold.py run --name omni3b-A --set train.variant=A --baseline
    python scripts/run_5fold.py run --name omni3b-v2 --dry-run          # show what would happen, spend nothing
    python scripts/run_5fold.py collect --name omni3b-v2                # only download + aggregate finished folds
    python scripts/run_5fold.py collect --runs omni3b-B-f0 omni3b-B5-f1 omni3b-B5-f2 omni3b-B5-f3 omni3b-B5-f4

`run` is safe to rerun at any point (e.g. after this machine drops off the network):
- a fold whose metrics.json is already on the runs Volume is skipped;
- a fold whose recorded call (runs/<name>.calls.json) is still running is reattached, not launched twice;
- any other fold is (re)spawned, and training resumes from its checkpoint if one exists.
Calls are spawned on the deployed app, so they run to completion whatever happens locally (see scripts/launch.py).
`--baseline` also runs the zero-shot baseline per fold (runs zs-<name>-f<k>), which is the comparison row.

Outputs: runs/<name>.log (all calls' logs; training lines carry [run] prefixes), runs/<name>/<run>/metrics.json per
fold, runs/<name>.summary.json and a printed table: per condition, mean and sample std over folds of the intended
question's accuracy/UAR/F1/ECE/NLL, the perceived question's KL/majority agreement, Noul Brier and intensity MAE.
`--results` appends the summary rows to RESULTS.md.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
APP, RUNS_VOLUME, CONDITIONS = "facedecide", "facedecide-runs", ("av", "face", "voice")
METRICS = [("intended", "acc"), ("intended", "uar"), ("intended", "f1_macro"), ("intended", "ece"), ("intended", "nll"),
           ("perceived", "kl"), ("perceived", "agree_majority"), ("noul", "brier"), ("intensity", "score_mae")]


# --- Modal helpers (imported lazily so `collect` aggregation and the tests don't need a Modal login)

def volume():
    import modal
    return modal.Volume.from_name(RUNS_VOLUME)


def remote_json(vol, path):
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:   # missing file (FileNotFoundError / modal NotFoundError)
        return None


def call_state(call_id):
    """'running' | 'done' | 'failed' for a recorded call."""
    import modal
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except Exception:
        return "failed"


# --- aggregation (pure)

def aggregate(per_fold):
    """per_fold: {run: metrics.json dict (its "test" section is used)} -> {condition: {question.metric: [mean, std, n]}}
    plus "selected_steps" and "temperatures" (per question: mean, std). std is the sample std (ddof=1; 0 for n=1)."""
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for q, m in METRICS:
            vals = [r["test"][q][cond][m] for r in per_fold.values() if cond in r.get("test", {}).get(q, {})]
            if vals:
                out[cond][f"{q}.{m}"] = [float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                         len(vals)]
    temps = {}
    for r in per_fold.values():
        for q, t in r.get("temperatures", {}).items():
            temps.setdefault(q, []).append(t)
    out["temperatures"] = {q: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] for q, v in temps.items()}
    out["selected_steps"] = {run: r.get("selected_step") for run, r in per_fold.items()}
    return out


def table(summary, label):
    cols = ["intended.acc", "intended.uar", "intended.f1_macro", "intended.ece", "intended.nll", "perceived.kl",
            "perceived.agree_majority", "noul.brier", "intensity.score_mae"]
    lines = [f"{label}: mean ± std over folds", "| cond | n | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for cond in CONDITIONS:
        row = summary.get(cond, {})
        n = max((v[2] for v in row.values()), default=0)
        cells = [f"{row[c][0]:.3f} ± {row[c][1]:.3f}" if c in row else "—" for c in cols]
        lines.append(f"| {cond} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- commands

def collect(runs, out_dir, label):
    vol = volume()
    per_fold, missing = {}, []
    for run in runs:
        m = remote_json(vol, f"{run}/metrics.json")
        if m is None:
            missing.append(run)
            continue
        per_fold[run] = m
        (out_dir / run).mkdir(parents=True, exist_ok=True)
        (out_dir / run / "metrics.json").write_text(json.dumps(m, indent=1))
    if missing:
        print(f"not finished (no metrics.json yet): {', '.join(missing)}")
    if not per_fold:
        return None
    summary = aggregate(per_fold)
    print(table(summary, f"{label} ({len(per_fold)} of {len(runs)} folds)"))
    return summary


def append_results(summary, label, config_hash, runs):
    lines = ["", f"5-fold summary `{label}` (config {config_hash}; runs {', '.join(runs)}): mean ± std over folds, test.",
             "", "| Cond | Folds | WAR | UAR | F1-macro | ECE (cal) | NLL (cal) | Perceived KL | Perceived agree |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        r = summary[cond]
        f = lambda k: f"{r[k][0]:.3f} ± {r[k][1]:.3f}"
        lines.append(f"| {cond} | {r['intended.acc'][2]} | {f('intended.acc')} | {f('intended.uar')} | {f('intended.f1_macro')} | "
                     f"{f('intended.ece')} | {f('intended.nll')} | {f('perceived.kl')} | {f('perceived.agree_majority')} |")
    with open(REPO / "RESULTS.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")


def run(a):
    from launch import follow
    folds = [int(k) for k in a.folds.split(",")]
    runs = [f"{a.name}-f{k}" for k in folds]
    jobs = [("train_fold", k, run_name, (k, run_name, a.set.split(), 0)) for k, run_name in zip(folds, runs)]
    if a.baseline:
        jobs += [("zero_shot", k, f"zs-{a.name}-f{k}", (k, f"zs-{a.name}-f{k}", True)) for k in folds]
    calls_path = REPO / "runs" / f"{a.name}.calls.json"
    calls = json.loads(calls_path.read_text()) if calls_path.exists() else {}
    if not a.dry_run and not a.no_deploy:
        subprocess.run([sys.executable, "-m", "modal", "deploy", str(REPO / "modal_app.py")], check=True, cwd=REPO)
    vol = volume()
    to_follow = []
    for fn, k, run_name, args in jobs:
        if remote_json(vol, f"{run_name}/metrics.json") is not None:
            print(f"{run_name}: done (metrics.json on the Volume); skipping")
            continue
        prev = calls.get(run_name)
        if prev and call_state(prev) == "running":
            print(f"{run_name}: call {prev} still running; reattaching")
            to_follow.append(prev)
            continue
        if a.dry_run:
            print(f"{run_name}: would spawn {fn}{args}" + (" (resuming its checkpoint)" if prev else ""))
            continue
        import modal
        call = modal.Function.from_name(APP, fn).spawn(*args)
        calls[run_name] = call.object_id
        calls_path.write_text(json.dumps(calls, indent=1))
        print(f"{run_name}: spawned {fn} as {call.object_id}")
        to_follow.append(call.object_id)
    if a.dry_run:
        return
    if to_follow and not a.no_follow:
        log = REPO / "runs" / f"{a.name}.log"
        print(f"following {len(to_follow)} call(s); logs -> {log}")
        follow(to_follow, str(log))
    elif to_follow:
        print("not following; rerun `run` (it reattaches) or use `collect` when done")
        return
    summary = collect(runs, REPO / "runs" / a.name, a.name)
    if summary is None:
        return
    config_hash = hashlib.sha256((REPO / "configs" / "omni_3b.yaml").read_bytes()).hexdigest()[:12]
    summary["config_hash"], summary["overrides"] = config_hash, a.set.split()
    (REPO / "runs" / f"{a.name}.summary.json").write_text(json.dumps(summary, indent=1))
    if a.baseline:
        base = collect([f"zs-{a.name}-f{k}" for k in folds], REPO / "runs" / a.name, f"zero-shot {a.name}")
        if base:
            (REPO / "runs" / f"zs-{a.name}.summary.json").write_text(json.dumps(base, indent=1))
    if a.results and all(summary[c].get("intended.acc", [0, 0, 0])[2] == len(folds) for c in CONDITIONS):
        append_results(summary, a.name, config_hash, runs)
        print("appended to RESULTS.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True, help="run prefix: runs are <name>-f0 .. <name>-f4")
    r.add_argument("--folds", default="0,1,2,3,4")
    r.add_argument("--set", default="", help="space-separated config overrides, e.g. 'train.variant=A train.epochs=4'")
    r.add_argument("--baseline", action="store_true", help="also run the zero-shot baseline per fold")
    r.add_argument("--results", action="store_true", help="append the summary to RESULTS.md when all folds finished")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-deploy", action="store_true", help="skip `modal deploy` (code unchanged since last deploy)")
    r.add_argument("--no-follow", action="store_true", help="launch and return")
    c = sub.add_parser("collect")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="prefix: collects <name>-f0 .. -f4")
    g.add_argument("--runs", nargs="+", help="explicit run names (e.g. folds launched under different names)")
    c.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        runs = a.runs or [f"{a.name}-f{k}" for k in range(5)]
        label = a.label or a.name or "runs"
        summary = collect(runs, REPO / "runs" / label, label)
        if summary:
            (REPO / "runs" / f"{label}.summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
