"""Start jobs on the deployed Modal app so they don't depend on this machine's connection, and follow them.

`modal run --detach` ties the call to the local client: when the network dropped, the client cancelled the
running training call on its way out (fold 0 was killed at step 600). Here the app is deployed once
(`modal deploy modal_app.py`) and jobs are spawned: a spawned call runs to completion whatever happens locally.

    modal deploy modal_app.py                                  # after every code change
    python scripts/launch.py train --fold 0 --name omni3b-B-f0 # resumes from the run's checkpoint if present
    python scripts/launch.py baseline --fold 1 --name zs-f1
    python scripts/launch.py follow runs/<name>.log <call_id> [<call_id> ...]  # stream logs until all calls end
"""
import argparse
import subprocess
import sys
import time

import modal

APP = "emotion-decider"


def spawn(fn, *args, **kwargs):
    call = modal.Function.from_name(APP, fn).spawn(*args, **kwargs)
    print(call.object_id)
    return call.object_id


def follow(call_ids, out):
    """Append the deployed app's logs (one stream for all its calls; training lines carry a [run] prefix) to `out`
    until every call finishes, reconnecting after drops. Each finished call appends "[launch] call ... finished:
    <result or error>"; at the end "exit 0" (all succeeded) or "exit 1" (what scratchpad watchers wait for)."""
    pending = {c: modal.FunctionCall.from_id(c) for c in call_ids}
    modal_cli = [sys.executable, "-m", "modal"]
    code, logs = 0, None
    with open(out, "a") as f:
        while pending:
            if logs is None or logs.poll() is not None:
                if logs is not None:
                    f.write("[launch] log stream ended; reconnecting\n"); f.flush()
                logs = subprocess.Popen(modal_cli + ["app", "logs", APP, "-f"], stdout=f, stderr=subprocess.STDOUT)
            for cid, call in list(pending.items()):
                try:
                    result = call.get(timeout=10)
                except TimeoutError:
                    continue
                except modal.exception.OutputExpiredError:
                    result = "result expired"
                except Exception as e:   # the call failed or was cancelled
                    result, code = f"{type(e).__name__}: {e}", 1
                f.write(f"[launch] call {cid} finished: {result}\n"); f.flush()
                del pending[cid]
        time.sleep(10)   # let the last log lines arrive
        logs.terminate()
        f.write(f"exit {code}\n"); f.flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--fold", type=int, default=0); t.add_argument("--name", required=True)
    t.add_argument("--set", default=""); t.add_argument("--max_steps", type=int, default=0)
    b = sub.add_parser("baseline"); b.add_argument("--fold", type=int, default=0); b.add_argument("--name", required=True)
    fo = sub.add_parser("follow"); fo.add_argument("out"); fo.add_argument("call_ids", nargs="+")
    a = ap.parse_args()
    if a.cmd == "train":
        spawn("train_fold", a.fold, a.name, a.set.split(), a.max_steps)
    elif a.cmd == "baseline":
        spawn("zero_shot", a.fold, a.name, True)
    else:
        follow(a.call_ids, a.out)


if __name__ == "__main__":
    main()
