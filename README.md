# Results

One row per run and condition. Intended question: WAR (= accuracy), UAR, ECE-15, NLL against the acted label.
Perceived question: KL and NLL against the condition's crowd vote distribution, agreement with the vote majority.
Calibrated = per-question temperatures fitted on the fold's dev actors. Config hash = first 12 hex of sha256 of
the config file used. Human raters (CREMA-D, our tally): intended accuracy voice 40.0%, face 57.4%, AV 63.1%.

| Date | Run | Config hash | Data / split | Cond | WAR | UAR | ECE (raw→cal) | NLL (cal) | Perceived KL (cal) | Perceived agree | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | zs-f0 (zero-shot Omni-3B) | 90d2d88c67fd | CREMA-D 5fold, fold 0 test (1553 clips) | av | 0.289 | 0.306 | 0.417→0.043 | 1.682 | 0.959 | 0.389 | T: intended 3.72, perceived 2.80, noul 3.26, intensity 3.88 |
| 2026-10-03 | zs-f0 | 90d2d88c67fd | fold 0 test | face | 0.317 | 0.333 | 0.337→0.036 | 1.668 | 0.828 | 0.459 | |
| 2026-10-03 | zs-f0 | 90d2d88c67fd | fold 0 test | voice | 0.246 | 0.263 | 0.389→0.052 | 1.758 | 0.685 | 0.603 | perceived voice ECE stays 0.172 calibrated |
| 2026-10-03 | omni3b-B-f0 (LoRA r16 + pointer head, variant B) | 90d2d88c67fd* | fold 0 test (1553 clips) | av | 0.835 | 0.835 | →0.022 | 0.449 | 0.323 | 0.784 | best dev ckpt step 2400/3912 (~3.7 ep); T: intended 1.42, perceived 0.98, noul 1.04, intensity 1.02 |
| 2026-10-03 | omni3b-B-f0 | 90d2d88c67fd* | fold 0 test | face | 0.655 | 0.657 | →0.034 | 0.868 | 0.435 | 0.701 | |
| 2026-10-03 | omni3b-B-f0 | 90d2d88c67fd* | fold 0 test | voice | 0.739 | 0.741 | →0.030 | 0.718 | 0.332 | 0.737 | perceived top-1 vs intended 0.402 (human voice raters 0.400) |

\* fold 0 trained with gradient checkpointing on for steps 0-900 and off afterwards (numerically equivalent); config
otherwise as `configs/omni_3b.yaml` at hash 90d2d88c67fd except `gradient_checkpointing`. Folds 1-4 use a 5-epoch
ceiling (and 8 clips per forward pass, numerically equivalent); fold 0 is kept as run (owner's decision): its selected
checkpoint (~3.7 epochs) is inside the 5-epoch window, but its cosine LR schedule spans 6 epochs, so at the selected
step its learning rate differs slightly from a 5-epoch run's. Dev-probe curve (loss):
0:1.43 200:1.03 400:0.99 600:0.95 1000:0.94 1200:0.93 1600:0.91 1800:0.90 2400:0.90 (best) 2800:0.96 3912:0.97.

| Date | Run | Config hash | Data / split | Cond | WAR | UAR | ECE (cal) | NLL (cal) | Perceived KL (cal) | Perceived agree | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | omni3b-B5-f1 | 0ab1a85ffda5 | fold 1 test (1470 clips) | av / face / voice | 0.818 / 0.605 / 0.713 | 0.818 / 0.604 / 0.714 | | | | | best step 2200/3315; T intended 1.27 |
| 2026-10-03 | omni3b-B5-f2 | 0ab1a85ffda5 | fold 2 test (1465 clips) | av / face / voice | 0.840 / 0.672 / 0.765 | 0.843 / 0.675 / 0.769 | | | | | best step 2600/3320; T intended 1.32 |
| 2026-10-03 | omni3b-B5-f3 | 0ab1a85ffda5 | fold 3 test (1470 clips) | av / face / voice | 0.823 / 0.616 / 0.783 | 0.824 / 0.612 / 0.785 | | | | | best step 2000/3315; T intended 1.22 |
| 2026-10-03 | omni3b-B5-f4 | 0ab1a85ffda5 | fold 4 test (1467 clips) | av / face / voice | 0.840 / 0.676 / 0.736 | 0.841 / 0.677 / 0.737 | | | | | best step 2200/3320; T intended 1.47 |

Folds 1-4: `configs/omni_3b.yaml` at 0ab1a85ffda5 (5-epoch ceiling, 8 clips per forward, no gradient checkpointing).
Per-fold full metrics: `runs/five-fold/<run>/metrics.json`.

5-fold summary `five-fold` (runs omni3b-B-f0, omni3b-B5-f1..f4; fold 0 as noted above): mean ± std over folds, test.

| Cond | Folds | WAR | UAR | F1-macro | ECE (cal) | NLL (cal) | Perceived KL | Perceived agree |
|---|---|---|---|---|---|---|---|---|
| av | 5 | 0.831 ± 0.010 | 0.832 ± 0.011 | 0.833 ± 0.011 | 0.030 ± 0.011 | 0.465 ± 0.027 | 0.338 ± 0.022 | 0.778 ± 0.017 |
| face | 5 | 0.645 ± 0.032 | 0.645 ± 0.035 | 0.642 ± 0.035 | 0.057 ± 0.016 | 0.924 ± 0.066 | 0.453 ± 0.035 | 0.680 ± 0.033 |
| voice | 5 | 0.747 ± 0.027 | 0.749 ± 0.028 | 0.748 ± 0.028 | 0.040 ± 0.008 | 0.706 ± 0.056 | 0.345 ± 0.012 | 0.731 ± 0.013 |

### Encoder-LoRA screen, fold 1 (2026-10-03)

`omni3b-B5enc-f1`: config 0ab1a85ffda5 + `lora.vision_targets=[attn.q,attn.k,attn.v,attn.proj,gate_proj,up_proj,down_proj,mlp.0,mlp.2]`
`lora.audio_targets=[q_proj,k_proj,v_proj,out_proj,fc1,fc2]` (r16 LoRA also on the ViT incl. merger and the audio encoder)
`train.clips_per_forward=4` (8 ran out of memory; numerically equivalent). Steps 0-400 used transformers' per-window
ViT attention loop, 400+ the batched equivalent (`omni.batch_vision_attention`; fp32 features identical). Best dev
step 1000/3315 (baseline 2200); T intended 1.23. Compared with omni3b-B5-f1 on the same 1470 test clips.

| Cond | WAR | UAR | ECE (cal) | NLL (cal) | Perceived KL | Perceived agree |
|---|---|---|---|---|---|---|
| av | 0.824 (base 0.818) | 0.822 | 0.025 (0.021) | 0.487 (0.508) | 0.347 (0.360) | 0.755 (0.763) |
| face | 0.638 (0.605) | 0.638 | 0.062 (0.065) | 0.980 (1.005) | 0.446 (0.478) | 0.668 (0.629) |
| voice | 0.769 (0.713) | 0.769 | 0.023 (0.048) | 0.647 (0.761) | 0.363 (0.344) | 0.685 (0.727) |

Per-class AV recall, base -> encoder LoRA (paired McNemar p): fear 0.657 -> 0.629 (0.35), sadness 0.700 -> 0.864
(<0.001), neutral 0.859 -> 0.732 (<0.001), happiness 0.956 -> 0.940 (0.39), anger 0.869 -> 0.889, disgust 0.869 -> 0.877.
The sadness gain is mostly a shift toward predicting sadness (share of AV predictions 17.9% -> 23.7%, sadness
precision 0.665 -> 0.619; acted neutral -> sadness 14 -> 35, fear -> sadness 59 -> 74). Single-modality accuracy
gains are significant: voice +5.6 (p<0.001), face +3.3 (p=0.014; face fear -10.8, face sadness +18.8).

### Encoder LoRA + augmentation + lower encoder LR, fold 1 (2026-10-03)

`omni3b-B5aug-f1`: config fede5b8167eb (= 0ab1a85ffda5 plus the new, default-off `augment` section and `lr_encoder`)
+ the encoder-LoRA overrides above + `train.clips_per_forward=4 train.lr_encoder=3.0e-5 augment.enabled=true`.
Best dev step 1800/3315 (dev loss 0.804; omni3b-B5enc-f1 0.825 at 1000, omni3b-B5-f1 0.834 at 2200); late-run
train/dev gap ~0.07 (encoder LoRA alone ~0.21). T intended 1.35. Same 1470 test clips as the other fold-1 runs.

| Cond | WAR | UAR | ECE (cal) | NLL (cal) | Perceived KL | Perceived agree |
|---|---|---|---|---|---|---|
| av | 0.837 (LM-only 0.818) | 0.835 | 0.020 (0.021) | 0.448 (0.508) | 0.310 (0.360) | 0.783 (0.763) |
| face | 0.667 (0.605) | | 0.017 (0.065) | 0.896 (1.005) | 0.413 (0.478) | 0.693 (0.629) |
| voice | 0.768 (0.713) | | 0.034 (0.048) | 0.655 (0.761) | 0.330 (0.344) | 0.711 (0.727) |

AV vs LM-only (paired McNemar p): overall +2.0 (0.033); fear 0.657 -> 0.741 (0.001), sadness 0.700 -> 0.716 (0.68),
disgust +6.3 (0.001), anger +4.0 (0.06), happiness 0.956 -> 0.964, neutral 0.859 -> 0.746 (<0.001). The sadness bias of
encoder LoRA alone is gone (sadness share of AV predictions 17.3%, LM-only 17.9%, enc 23.7%); neutral stays
under-predicted (11.2% of predictions vs 14.4% acted; acted neutral -> sadness 21, anger 16, fear 10). Face +6.2 and
voice +5.5 (both p<0.001).

5-fold summary `omni3b-B5aug` (config fede5b8167eb; runs omni3b-B5aug-f0, omni3b-B5aug-f1, omni3b-B5aug-f2, omni3b-B5aug-f3, omni3b-B5aug-f4): mean ± std over folds, test.

| Cond | Folds | WAR | UAR | F1-macro | ECE (cal) | NLL (cal) | Perceived KL | Perceived agree |
|---|---|---|---|---|---|---|---|---|
| av | 5 | 0.853 ± 0.011 | 0.854 ± 0.013 | 0.853 ± 0.012 | 0.026 ± 0.008 | 0.410 ± 0.024 | 0.302 ± 0.014 | 0.805 ± 0.019 |
| face | 5 | 0.666 ± 0.019 | 0.666 ± 0.022 | 0.664 ± 0.022 | 0.061 ± 0.030 | 0.878 ± 0.031 | 0.405 ± 0.022 | 0.715 ± 0.021 |
| voice | 5 | 0.788 ± 0.017 | 0.790 ± 0.018 | 0.790 ± 0.018 | 0.038 ± 0.019 | 0.604 ± 0.051 | 0.333 ± 0.007 | 0.739 ± 0.019 |

Per fold AV accuracy, LM-only (five-fold) -> omni3b-B5aug: f0 0.835 -> 0.849, f1 0.818 -> 0.837, f2 0.840 -> 0.868,
f3 0.823 -> 0.855, f4 0.840 -> 0.856 (better on every fold). Pooled over the 5 folds (7425 test clips, paired McNemar):
AV +2.2 (+442 / -282, p<0.001); fear 0.722 -> 0.775, disgust 0.814 -> 0.869 (both p<0.001), anger +1.4, neutral +1.8,
happiness +0.9, sadness 0.725 -> 0.706 (p=0.12). Face +2.2, voice +4.1 (both p<0.001). Fold 1's neutral drop (-11.3)
does not recur in the pool. Selection caveat: the recipe was chosen from fold-1 test results (two screens); folds 0, 2,
3, 4 alone give AV 0.857, so the headline does not rest on fold 1. Fold 0 of the LM-only row used the 6-epoch schedule.
