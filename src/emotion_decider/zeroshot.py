"""Zero-shot baseline: the untrained Omni thinker answering the same questions by its letter logits.

Each question becomes the rest of a chat turn after the clip — instructions, lettered options, "Answer with the
letter..." — then `<|im_end|>\\n<|im_start|>assistant\\n`; the option logits are the vocabulary logits of the
letters A, B, ... at that last position. All questions about a clip share one packed pass (same block mask and
positions as the decision model), so this costs what the decision model costs.

    python -m emotion_decider.zeroshot --config configs/omni_3b.yaml --fold 0 --out runs/zs-f0 [--calibrate]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import metrics as M
from .model import Branch
from .omni import packed_hidden_batch
from .questions import option_texts, user_tokens
from .train import fit_temperatures, load_config, load_rows, predict, splits, spread, summarize

TAIL = "<|im_end|>\n<|im_start|>assistant\n"
LETTERS = "ABCDEFGHIJ"


class ZeroShot:
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    suffix = ""   # each question continues the user turn

    def __init__(self, thinker, tokenizer):
        self.thinker, self.tokenizer = thinker.eval(), tokenizer
        self.letters = []
        for L in LETTERS:
            ids = tokenizer(L, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise ValueError(f"letter {L} is not a single token")
            self.letters.append(ids[0])
        self.tail = tokenizer(TAIL, add_special_tokens=False).input_ids

    @property
    def device(self):
        return next(self.thinker.parameters()).device

    def branch(self, question):
        options = "\n".join(f"{L}. {t}" for L, t in zip(LETTERS, option_texts(question)))
        body = f"\n{question['instructions']}\nOptions:\n{options}\nAnswer with the letter of one option only."
        ids = user_tokens(self.tokenizer, body) + self.tail
        return Branch(ids=ids, decide=len(ids) - 1, opts=[])

    @torch.no_grad()
    def logits_batch(self, clips, question_lists):
        branch_lists = [[self.branch(q) for q in qs] for qs in question_lists]
        hidden = packed_hidden_batch(self.thinker, [c.to(self.device) for c in clips], branch_lists)
        return [[self.thinker.lm_head(h[-1])[self.letters[: len(option_texts(q))]].float() for h, q in zip(hs, qs)]
                for hs, qs in zip(hidden, question_lists)]

    def logits(self, clip, questions):
        return self.logits_batch([clip], [questions])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate", action="store_true", help="also fit per-question temperatures on dev")
    ap.add_argument("--limit_clips", type=int)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    b = cfg["backbone"]
    processor = AutoProcessor.from_pretrained(b["name"], revision=b.get("revision"))
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        b["name"], revision=b.get("revision"), dtype=getattr(torch, b["dtype"]), attn_implementation=b.get("attn", "sdpa"))
    from .omni import batch_vision_attention
    batch_vision_attention(thinker)
    scorer = ZeroShot(thinker.to("cuda" if torch.cuda.is_available() else "cpu"), processor.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root, rows = cfg["data"]["root"], load_rows(cfg["data"]["root"])
    sp = splits(cfg, a.fold, rows)
    if a.limit_clips:   # a sample across actors, as the trained model's short evaluations use
        sp = {k: spread(v, a.limit_clips, f"eval/{cfg['seed']}/{a.fold}/{k}") for k, v in sp.items()}
    results, workers, bs = {}, cfg["data"].get("num_workers", 0), cfg["train"].get("eval_batch", 1)
    test = predict(scorer, processor, rows, sp["test"], root, num_workers=workers, batch_size=bs)
    results["test_raw"] = summarize(test, rows)
    if a.calibrate:
        temps = fit_temperatures(predict(scorer, processor, rows, sp["dev"], root, num_workers=workers, batch_size=bs))
        for p in test:
            p["probs"] = M.softmax(np.array(p["logits"]), temps.get(p["question"], 1.0)).tolist()
        results["temperatures"], results["test_calibrated"] = temps, summarize(test, rows)
    with open(out / "predictions_test.jsonl", "w") as f:
        for p in test:
            f.write(json.dumps(p) + "\n")
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
