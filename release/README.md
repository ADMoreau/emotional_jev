---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE
base_model: Qwen/Qwen2.5-Omni-3B
pipeline_tag: video-classification
tags:
  - emotion-recognition
  - audio-visual
  - calibration
  - crema-d
  - peft
  - lora
  - built-with-qwen
---

# EmotionDecider-AV-3B

**Built with Qwen.** LoRA adapters and a decision head on top of
[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) (commit `f75b40e3`), for non-commercial research.

EmotionDecider reads a short clip of a face and/or voice and answers typed questions about the emotion shown, as
calibrated probabilities over each question's answer options. All questions are scored in **one prefill pass**, with
no text generation. It was trained and evaluated only on **CREMA-D**, a dataset of acted, lab-recorded clips.

Code, training recipe and full results: https://github.com/ADMoreau/emotional_jev
<!-- TODO: update the link if the GitHub repo is renamed. -->

## What it does

Input: a clip of up to a few seconds: 16 face-crop frames (224 px) and/or 16 kHz mono audio. Either modality may be
missing (face-only or voice-only). Questions use a Choice / Noul / Score schema (a "System One"-style,
Jev-compatible interface; this project is not affiliated with TypeSafe AI):

| Type | Example | Output |
|---|---|---|
| Choice | "Which emotion was the actor portraying?" over 6 options | a probability per option, plus confidence |
| Noul | "Would viewers say this person sounds angry?" | P(yes) |
| Score | "How strongly does emotion come through?" over 4 ordered levels | a probability per level, expected level, confidence |

Each question is an isolated branch over the shared clip state, so an answer never depends on which other questions
were asked in the same pass. The model was trained on four question families: the acted emotion (6 classes), how most
viewers would read the clip (crowd vote distribution), per-emotion yes/no questions, and perceived intensity.
Questions far outside these families are untested.

## Files

- `decision.safetensors` (222 MB): LoRA adapters (language model, vision encoder, audio encoder; rank 16) and the pointer head.
  The base weights are not included; they are loaded from the pinned Qwen2.5-Omni-3B commit.
- `decision.json`: adapter configuration and training metadata.
- `calibration.json`: per-question temperatures fitted on held-out actors.
- `LICENSE` (Qwen Research License Agreement) and `NOTICE`.

## Usage

Install the code (`pip install git+https://github.com/ADMoreau/emotional_jev`), download this repository, then:

```python
from emotion_decider.inference import load, answer

model = load("EmotionDecider-AV-3B")   # fetches the pinned Qwen2.5-Omni-3B base on first use; GPU if available
emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
questions = {
    "intended": {"type": "choice", "instructions": "Which emotion was the actor portraying?",
                 "criteria": {e: None for e in emotions}},
    "perceived": {"type": "choice", "instructions": "How would most viewers read this clip?",
                  "criteria": {e: None for e in emotions}},
    "angry": {"type": "noul", "instructions": "Would viewers say this person sounds angry?"},
    "intensity": {"type": "score", "instructions": "How strongly does emotion come through?",
                  "criteria": ["no clear emotion", "mild", "clear", "intense"]},
}
out = answer(model, questions, frames=frames, fps=fps, audio=audio)
out["answers"]["intended"]   # {"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}
```

`frames` is a `[T, H, W, 3]` uint8 array of face crops (training used 16 frames at 224 px, a square box about 1.3x
the detected face; `scripts/extract_faces.py` with `emotion_decider/faces.py` is the training pipeline), `fps` the rate at which those
frames were sampled, and `audio` float32 16 kHz mono over the same time span. Pass only `frames` (with `fps`) or only
`audio` for face-only or voice-only input. Question ids `intended`, `perceived`, `noul` and `intensity` use the
temperatures fitted for those question families; other ids use their type's default.

## Results

Speaker-independent evaluation: 5 actor-disjoint folds of CREMA-D (18-19 test actors per fold), each fold trained
with the same recipe as this model; mean ± std over folds. The released model is trained on all 91 actors (8 held out
for checkpoint selection and calibration), so it has no test set of its own; these cross-validation numbers are its
expected performance. (On its 8 dev actors it scores 89.1% with face + voice, but those actors also chose the
checkpoint and fitted the temperatures, so that number is optimistic.)

**Acted emotion (6 classes)**

| Input | Accuracy | UAR | Macro-F1 | ECE (calibrated) |
|---|---|---|---|---|
| Face + voice | **85.3 ± 1.1** | 85.4 ± 1.3 | 85.3 ± 1.2 | 0.026 |
| Face only | 66.6 ± 1.9 | 66.6 ± 2.2 | 66.4 ± 2.2 | 0.061 |
| Voice only | 78.8 ± 1.7 | 79.0 ± 1.8 | 79.0 ± 1.8 | 0.038 |

Per class (face + voice, pooled over folds): anger 91.6, disgust 86.9, fear 77.5, happiness 96.3, neutral 89.3,
sadness 70.6. Most errors are among sadness, fear and neutral, which human raters also confuse.

For comparison: an individual CREMA-D rater identifies the acted emotion 63.1% of the time with face and voice, 57.4%
from the face, 40.0% from the voice. Published speaker-independent results: HiCMAE 84.9%, DE-III 83.7% (F1-micro);
fold assignments differ between papers, so differences under about a point are not meaningful. Numbers reported on
random clip-level splits (e.g. 89.49% on EmoBox fold 1) share actors between train and test and are not comparable.

**How viewers would read the clip (calibration against human perception)**

The target is the CREMA-D crowd's vote distribution for the same input condition (face + voice votes for face + voice
input, face-only votes for face-only input, voice-only votes for voice-only input).

| Input | KL to crowd votes | Top answer = crowd majority |
|---|---|---|
| Face + voice | 0.30 | 80.5% |
| Face only | 0.41 | 71.5% |
| Voice only | 0.33 | 73.9% |

**By actor group** (face + voice, acted emotion, pooled 5-fold test predictions; n = clips, actors in brackets):

| Group | Model accuracy | Human rater accuracy |
|---|---|---|
| Female (43) / Male (48) | 88.7 / 82.3 | 66.0 / 60.2 |
| Age 20-29 (34) / 30-44 (32) / 45-59 (19) / 60+ (6) | 87.0 / 85.8 / 84.2 / 76.8 | 63.6 / 62.1 / 64.8 / 57.4 |
| Caucasian (61) / African American (22) / Asian (7) | 86.5 / 81.8 / 86.6 | 63.1 / 63.1 / 61.8 |
| Hispanic (10) / Not Hispanic (81) | 88.5 / 84.9 | |

The sex and age gaps are also present, in smaller form, for human raters. **The gap for African American actors
(-4.7 points against Caucasian actors) is not present for human raters**, so it is specific to the model; the training
data has 22 African American actors against 61 Caucasian actors. Groups with few actors (60+, Asian) have wide
uncertainty.

## Training

- Data: CREMA-D (7,442 clips, 91 actors, 12 sentences, 6 emotions; 17 clips excluded for audio/video defects).
  Labels: the acted emotion and the crowd's vote distributions (2,443 raters; face-only, voice-only and face + voice).
- Recipe: rank-16 LoRA on the language model, the vision encoder and the audio encoder, plus a pointer head over
  option tokens. Each training clip is shown face + voice, face only or voice only with the matching crowd votes as
  the perception target. Training-time augmentation: crop, flip and colour jitter on frames; speed, gain and noise on
  audio. Soft-label cross-entropy, up to 5 epochs, checkpoint chosen by dev loss, then per-question temperature
  scaling on 8 held-out actors.
- Compute: one A100-80GB, about 2.5 hours.

## Intended use and limitations

- **Intended use:** research on audio-visual emotion recognition, multimodal fusion, and calibration against human
  perception, on acted clips similar to CREMA-D.
- **Out of scope:** any decision about real people, including workplace, hiring, education, healthcare, security,
  surveillance, policing or border use; identification of individuals; commercial use (the Qwen license is
  non-commercial). Many jurisdictions restrict emotion recognition (the EU AI Act prohibits it in workplaces and
  education).
- **What the labels mean:** the acted label is the emotion an actor was asked to portray; the perception labels are
  how crowd raters read the clip. Neither measures what anyone felt. Outputs describe expression, not internal state.
- **Acted, lab-recorded data only:** front-facing actors, fixed background and lighting, scripted English sentences.
  No claim is made about spontaneous expressions, other languages, other cultures, or in-the-wild video.
- **Input assumptions:** face crops like the training data's (a detected face, square box about 1.3x the face, 224 px),
  and audio covering the same time span as the frames.
- **Known weaknesses:** sadness and fear are the least accurate classes; face-only accuracy is well below voice-only;
  the demographic gaps above.

## Licenses and attribution

- Weights: derivative of Qwen2.5-Omni-3B under the Qwen Research License Agreement (non-commercial). Qwen is licensed
  under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.
- Training data: CREMA-D, Open Database License (ODbL). Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova,
  A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Affective
  Computing, 5(4), 377-390. This release contains no CREMA-D media or labels.
- Code: Apache-2.0; the packed decision path derives from Kev (https://github.com/jaredpalmer/kev).
