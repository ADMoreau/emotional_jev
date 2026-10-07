"""Jev-format questions and targets for a CREMA-D clip, and their token encoding.

Per clip and condition (the modality the model is shown: av, face or voice):
- intended  Choice  which emotion the actor was asked to portray; target = one-hot acted label (accuracy question)
- perceived Choice  how viewers read the clip; target = that condition's crowd vote distribution (calibration)
- noul      Noul    on a fraction of clips: would a typical viewer name one emotion; target = its vote share
- intensity Score   on a fraction of clips: how strongly emotion comes through; target = raters' intensity,
                    binned (a rater who chose neutral counts as "no clear emotion": their 1-100 rating scores
                    how neutral the clip seemed, mean 66, not how emotional)

Phrasings, Choice option order and whether options carry descriptions are drawn from `rng`, so a training
loop can redraw them every epoch. Questions follow the API schema (type / instructions / criteria) verbatim.
"""
import re

from .crema import EMOTIONS
from .model import Branch

CONDITIONS = ("av", "face", "voice")
NAMES = list(EMOTIONS.values())      # anger, disgust, fear, happiness, neutral, sadness
CODE = {v: k for k, v in EMOTIONS.items()}
DESCRIPTIONS = {
    "anger": "irritation, hostility or rage",
    "disgust": "revulsion, distaste or contempt",
    "fear": "worry, alarm or terror",
    "happiness": "pleasure, amusement or joy",
    "neutral": "no particular emotion; calm and matter-of-fact",
    "sadness": "disappointment, sorrow or grief",
}
ADJECTIVES = {"anger": "angry", "disgust": "disgusted", "fear": "afraid", "happiness": "happy",
              "neutral": "emotionally neutral", "sadness": "sad"}
INTENDED = [
    "Which emotion is the actor portraying in this clip?",
    "The speaker was asked to act out one emotion. Which one?",
    "What emotion was this person instructed to express?",
    "Identify the emotion the actor intended to convey.",
]
PERCEIVED = [
    "How would a typical viewer read this person's emotion?",
    "Which emotion would an ordinary observer say this person is showing?",
    "If someone were shown this clip, which emotion would they most likely name?",
    "What emotion does this person come across as expressing?",
]
NOUL = [
    "Would a typical viewer say this person seems {adj}?",
    "Does this person come across as {adj}?",
    "Would an ordinary observer describe the emotion here as {name}?",
]
INTENSITY = [
    "How strongly does emotion come through in this clip?",
    "How intense is the emotion this person expresses?",
]
LEVELS = [
    "No clear emotion comes through; the person seems neutral or flat.",
    "An emotion is present but mild and easy to miss.",
    "The emotion is clearly visible or audible.",
    "The emotion is intense and unmistakable.",
]
LEVEL_EDGES = (40, 70)               # non-neutral 1-100 ratings: <=40 mild, <=70 clear, else intense


def intensity_distribution(ratings):
    """[p0..p3] over LEVELS from [[emotion code, level | None], ...]; ratings without a level are skipped."""
    counts = [0] * len(LEVELS)
    for emo, level in ratings:
        if emo == "N":
            counts[0] += 1
        elif level is not None:
            counts[1 if level <= LEVEL_EDGES[0] else 2 if level <= LEVEL_EDGES[1] else 3] += 1
    n = sum(counts)
    return [c / n for c in counts] if n else None


def _choice(rng, instructions, dist):
    names = NAMES[:]
    rng.shuffle(names)
    described = rng.random() < 0.5
    q = {"type": "choice", "instructions": instructions,
         "criteria": {n: DESCRIPTIONS[n] if described else None for n in names}}
    return q, {"probabilities": {n: dist[n] for n in names}}


def make_questions(row, condition, rng, p_noul=0.4, p_score=0.5):
    """(questions, targets) for one targets.jsonl row shown under `condition`."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    perceived = row["perceived"][condition]
    qs, ts = {}, {}
    intended = {n: float(CODE[n] == row["intended"]) for n in NAMES}
    qs["intended"], ts["intended"] = _choice(rng, rng.choice(INTENDED), intended)
    qs["perceived"], ts["perceived"] = _choice(rng, rng.choice(PERCEIVED), perceived["distribution"])
    if rng.random() < p_noul:
        # half the time ask about the acted emotion, otherwise any other: keeps "yes" targets from being rare
        name = EMOTIONS[row["intended"]] if rng.random() < 0.5 else rng.choice([n for n in NAMES if CODE[n] != row["intended"]])
        q = {"type": "noul", "instructions": rng.choice(NOUL).format(adj=ADJECTIVES[name], name=name)}
        if rng.random() < 0.5:
            q["criteria"] = {"true": f"A typical viewer would call the person {ADJECTIVES[name]}.",
                             "false": f"A typical viewer would name some other emotion, or none."}
        share = perceived["distribution"][name]
        qs["noul"], ts["noul"] = q, {"probabilities": {"true": share, "false": 1 - share}}
    dist = intensity_distribution(perceived["ratings"])
    if dist is not None and rng.random() < p_score:
        qs["intensity"] = {"type": "score", "instructions": rng.choice(INTENSITY), "criteria": LEVELS[:]}
        ts["intensity"] = {"probabilities": {str(i): p for i, p in enumerate(dist)}}
    return qs, ts


# --- token encoding: [<q> instructions] then per option [<opt> text </opt>], then [<decide>]

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Caller text can never produce control tokens: `<|name|>` is defanged before tokenizing (from Kev)."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def option_keys(question):
    """Option keys in the order they are presented (and scored)."""
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def option_texts(question):
    if question["type"] == "noul":
        c = question.get("criteria") or {}
        return [f"yes: {c['true']}" if c.get("true") else "yes", f"no: {c['false']}" if c.get("false") else "no"]
    if question["type"] == "score":
        return list(question["criteria"])
    return [k if d is None else f"{k}: {d}" for k, d in question["criteria"].items()]


def encode_question(tok, question, delimiters):
    """-> Branch for one question. delimiters = token ids of (question, option, end_option, decide)."""
    q_id, o_id, c_id, d_id = delimiters
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        import json
        instructions = json.dumps(instructions)
    ids, opts = [q_id] + user_tokens(tok, instructions), []
    for text in option_texts(question):
        ids += [o_id] + user_tokens(tok, text) + [c_id]
        opts.append(len(ids) - 1)
    ids.append(d_id)
    return Branch(ids=ids, decide=len(ids) - 1, opts=opts)


def delimiter_ids(tok):
    from .omni import DECIDE, END_OPTION, OPTION, QUESTION
    ids = tuple(tok.convert_tokens_to_ids(t) for t in (QUESTION, OPTION, END_OPTION, DECIDE))
    if any(i is None or i == tok.unk_token_id for i in ids):
        raise ValueError("tokenizer lacks the delimiter tokens")
    return ids
