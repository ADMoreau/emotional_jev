# Fork vs. build: Kev / Qev / Valen

> **2026-10-02 update:** the backbone is now Qwen2.5-Omni-3B (attention-only, audio + video), not
> Qwen3.5. Kev's packed block-causal path applies directly; the DeltaNet fork below is only needed
> for hybrid backbones. Qev/Valen's Qwen3.5 loading code no longer carries over. The rest of this
> review stands as written for Qwen3.5.

Reviewed 2026-10-02 at kev `84847f0` (2026-10-01), qev `0af3c10` (2026-09-26), Valen `33de9f9` (2026-09-30).
All three are Apache-2.0.

## Recommendation

**Build a small package of our own. Vendor Kev's decision core and copy video plumbing from Qev/Valen.**
Don't fork any of the three whole. Each fork would mean either rewriting its core forward path or carrying
a lot of code we don't need.

## What each repo gives us

| Need (from CLAUDE.md) | Kev | Qev | Valen |
|---|---|---|---|
| Video in | No (text-only, `AutoModelForCausalLM.model` on `-Base`) | Yes: full `Qwen3_5ForConditionalGeneration` + processor, explicit frames/fps | Yes: processor + chat template, frame metadata kept |
| State computed once, branches fork DeltaNet state | **Yes**: `kev/shared_prefix.py` (training, grads flow into state) + prefix cache (serving), with parity tests | No: `question_isolation: "independent_rows"`, enforced at checkpoint load | No: `use_cache=False`, every branch a full forward |
| Soft-label CE | Yes (`q["target"]`) | Not found | Yes (+ RPS, Brier) |
| Temperature calibration | Yes: `scripts/calibrate_checkpoint.py`, `temperature_groups.py` | Calibration split, scalar T | Scalar T at inference only; eval CLI ignores it |
| Pointer head | `<decide>` · `</opt>`, dp=256 | Same | Same, but Score = one branch **per level** |
| Train merger + top-4 ViT | No | No (vision frozen) | **Yes**: `stage: vision_top` |
| Correctness tests | Rows == packed, shared prefix == rows (fp32 ~1e-5), prefix cache == full pass | Frozen-base hashes, MLX parity | Architecture contracts |
| transformers pin | `>=5.17,<6` | `>=5.17,<5.18` | `==5.4.0`, torch 2.6 |
| Core size (excluding scripts) | ~19k lines (lots of suite/benchmark/MLX/CUDA-graph code) | ~12k (Snake, LFM2, MLX, Homebrew, web UI) | ~2.9k core + 5k eval |

## Why not fork

- **Kev** has the hard part done: forking the recurrent state per branch, with gradients. But it's text-only.
  Adding video touches `encode`, `prefix` and `shared_prefix`, which is most of the core. The rest of
  the repo (suites, rounds, MLX, CUDA graphs, Modal) is unrelated to our work.
- **Qev** loads video correctly, but by design it recomputes state per question (format-checked). With
  ~3 questions/clip that's ~3× the backbone compute *and* the ViT runs once per question. It has no
  soft-label path, and its video decisions are untrained.
- **Valen** matches our features best (soft CE, `vision_top`, video), but it has the same per-branch
  recompute, eager attention only, an old transformers pin, and its Score-per-level branching differs
  from our design. We'd have to replace its forward path anyway.

## Plan

Lift into `src/` with attribution (ship the Apache LICENSE, add a NOTICE entry, and mark modified files):

1. **From Kev:** `encode`/`rows_of` layout, `PointerHead`, `shared_prefix.Prefix` + `branch_hidden`,
   `question_loss` soft-target path, and the temperature-fitting script. Port `test_shared_prefix_matches_rows`
   and `test_prefix_cache_matches_full_pass` as our correctness tests (b).
2. **From Qev:** `_load_foundation` (full `Qwen3_5ForConditionalGeneration`, strict load check) and the
   video branch of `tokenization.py` (frames as `[T,H,W,C]` arrays + explicit fps, no path opening).
3. **From Valen:** parameter-stage freezing (`warmup`/`text`/`joint`/`vision_top`) for the ViT ablation, and
   RPS as the optional ordinal loss for Score.

## The main new engineering: MRoPE

Qwen3.5 video tokens use 3-D multimodal RoPE (`get_rope_index` → `position_ids` [3, B, L] + `rope_deltas`).
Kev's encoder emits 1-D positions. In the shared-prefix path, we need to:
- Run the vision tower + merger once on the state and build `inputs_embeds`.
- Compute 3-D positions for the state via `get_rope_index`. Each branch's text positions continue from
  `max(state positions) + 1` on all three axes, and every branch restarts at the same offset.
- Keep `Prefix`'s conv-window/recurrent-state handling unchanged. It doesn't depend on positions.

Verify this with a tiny-config Qwen3.5-VL test: shared-prefix logits must equal one independent row per
question (the Qev/Valen approach) within fp32 tolerance. That gives us a reference implementation for free.

## CLAUDE.md wording to fix

Fixed 2026-10-02. A block-causal mask alone can't isolate branches on Qwen3.5 because DeltaNet layers
ignore attention masks. The fix is to fork the recurrent/conv state per branch inside each DeltaNet layer.
This is still one forward pass in training (Kev's shared prefix); serving may split it into state + branches
so the state can be cached.
