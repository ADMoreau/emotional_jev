# facedecide

An open audio-visual **decision model** for emotion recognition. Given a short clip (face video and/or voice) and
any number of typed questions, it returns calibrated probabilities over each question's answer options in **one
prefill pass**, with no text generation. v1 is trained and evaluated on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
only.

The request/response format follows the "System One" decision-model interface pattern (Choice, Noul and Score
questions; a Jev-compatible interface). This project is independent and not affiliated with TypeSafe AI.

**Status:** v1 trained and evaluated. Weights and model card are not released yet.

## Results (v1)

CREMA-D, 5 actor-disjoint folds (speaker-independent; 18-19 test actors per fold), mean ± std over folds.
The model is asked which emotion the actor was portraying (6 classes), with face and voice together (AV), face
only, or voice only.

| Input | Accuracy | UAR | ECE (calibrated) | NLL |
|---|---|---|---|---|
| Audio-visual | **85.3 ± 1.1** | 85.4 ± 1.3 | 0.026 | 0.410 |
| Face only | 66.6 ± 1.9 | 66.6 ± 2.2 | 0.061 | 0.878 |
| Voice only | 78.8 ± 1.7 | 79.0 ± 1.8 | 0.038 | 0.604 |

For reference, CREMA-D's individual raters identify the acted emotion 63.1% of the time (AV), 57.4% (face) and
40.0% (voice). Published speaker-independent results: HiCMAE 84.9%, DE-III 83.7% (F1-micro); fold assignments
differ between papers, so differences under about a point are not meaningful. Results on random clip-level splits
(e.g. 89.49% on EmoBox fold 1) put the same actors in train and test and are not comparable.

**Calibration against human perception.** A second question asks how most viewers would read the clip; its target is
the crowd's vote distribution for the same input condition. KL to the vote distribution: 0.30 (AV), 0.41 (face),
0.33 (voice); the model's top answer matches the crowd majority on 80.5%, 71.5% and 73.9% of clips.

Every run, ablation and caveat (including how the recipe was selected) is logged in [RESULTS.md](RESULTS.md).

## How it works

- **Backbone:** [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), thinker only. Face crops (16 frames,
  224 px) and 16 kHz audio share one timeline (TMRoPE).
- **One pass, many questions:** the sequence is `[clip state][question 1 branch][question 2 branch]...`. A
  block-causal mask lets each branch attend to the clip and itself only, so an answer never depends on which other
  questions were asked, and all branches are scored in one packed prefill. Derived from
  [Kev](https://github.com/jaredpalmer/kev).
- **Pointer head:** a query from the `<decide>` token is scored against keys from each option's end token; softmax
  over options. Noul is a two-option pointer; Score uses ordered options.
- **Training:** rank-16 LoRA on the language model, the ViT and the audio encoder, plus the head. Each clip is shown
  audio-visual, face-only or voice-only (modality dropout) with the matching crowd votes as the perceived target;
  training-time augmentation (crop/flip/colour jitter, audio speed/gain/noise). Soft-label cross-entropy, then
  per-question temperature scaling on held-out dev actors.
- **Questions per clip:** intended emotion (Choice, acted label), perceived emotion (Choice, crowd votes), "would
  viewers say this person looks/sounds X?" (Noul), and perceived intensity (Score, 4 levels).

## Repository layout

```
src/facedecide/   omni.py (packed multi-branch prefill), decision.py (LoRA + pointer head), train.py,
                  questions.py, augment.py, dataset.py, metrics.py, zeroshot.py, faces.py, crema.py
                  model.py: earlier Qwen3.5 (hybrid DeltaNet) path, kept as a reference
scripts/          data preparation, Modal launchers, 5-fold runner
configs/          omni_3b.yaml; splits/ (fold assignments, exclusion list: IDs only)
tests/            CPU tests on tiny random models and real clips (packing parity, isolation, training smoke tests)
modal_app.py      cloud jobs (data preparation, training, zero-shot baseline)
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.lock
.venv/bin/python -m pytest                       # CPU tests (data-dependent ones skip without CREMA-D)
```

Data (CREMA-D via Git LFS, then targets, audio alignment, face crops, audit, manifests):

```bash
python scripts/build_crema_targets.py && python scripts/audio_offsets.py && python scripts/extract_faces.py
python scripts/audit_crema.py && python scripts/build_manifests.py --seed 0
```

Training and evaluation run on Modal (one A100-80GB per fold, about 2.5 h each). The v1 recipe:

```bash
modal run modal_app.py::data && modal run modal_app.py::pack
python scripts/run_5fold.py run --name omni3b-B5aug --results --set "\
lora.vision_targets=[attn.q,attn.k,attn.v,attn.proj,gate_proj,up_proj,down_proj,mlp.0,mlp.2] \
lora.audio_targets=[q_proj,k_proj,v_proj,out_proj,fc1,fc2] \
train.clips_per_forward=4 train.lr_encoder=3.0e-5 augment.enabled=true"
```

## Data, licenses and intended use

- **Code:** Apache-2.0 ([LICENSE](LICENSE)); portions derived from Kev, Qev and Valen (see [NOTICE](NOTICE)).
- **Weights** (when released): derived from Qwen2.5-Omni-3B and under the Qwen Research License, non-commercial
  research only.
- **CREMA-D:** Open Database License (ODbL), Cao et al. 2014. This repository holds no media, frames, audio or
  labels, only clip/actor IDs and code.
- **Scope:** research on acted, lab-recorded clips. Labels describe the emotion an actor was asked to portray and how
  raters read it, not what anyone felt. Not for workplace, education, surveillance or identification uses, and no
  claims about in-the-wild performance.
