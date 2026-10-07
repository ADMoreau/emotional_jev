"""CREMA-D labels: clip names, per-modality crowd ratings, and actor-disjoint folds.

Clip names are ACTOR_SENTENCE_EMOTION_LEVEL, e.g. 1001_DFA_ANG_XX (intended emotion and acted intensity).
finishedResponses.csv has one row per rating: queryType 1 = voice-only, 2 = face-only, 3 = audio-visual;
respEmo is the perceived emotion (A/D/F/H/N/S) and respLevel the rater's perceived intensity (1-100).
"""
import csv
import random
from collections import defaultdict

EMOTIONS = {"A": "anger", "D": "disgust", "F": "fear", "H": "happiness", "N": "neutral", "S": "sadness"}
CODES = {"ANG": "A", "DIS": "D", "FEA": "F", "HAP": "H", "NEU": "N", "SAD": "S"}
MODALITIES = {"1": "voice", "2": "face", "3": "av"}
ACTED_LEVELS = ("LO", "MD", "HI", "XX")   # XX = unspecified (only sentence IEO varies the level)


def parse_clip(name):
    actor, sentence, emotion, level = name.split("_")
    if emotion not in CODES or level not in ACTED_LEVELS:
        raise ValueError(f"not a CREMA-D clip name: {name}")
    return {"actor": actor, "sentence": sentence, "intended": CODES[emotion], "acted_level": level}


def distracted(emo_path, limit_ms=10000):
    """(session, queryType, questNum) of ratings whose first click came over 10 s after the clip, from
    finishedEmoResponses.csv. The official tabulation (processFinishedResponses.R) drops these."""
    out = set()
    with open(emo_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                slow = float(r["ttr"]) > limit_ms
            except ValueError:            # NA: R's as.numeric gives NA, which which() skips
                continue
            if slow:
                out.add((r["sessionNums"], r["queryType"], r["questNum"]))
    return out


def load_ratings(path, emo_path=None):
    """-> {clip: {modality: {"votes": {code: count}, "ratings": [[code, respLevel | None], ...]}}} over every clip
    and modality.
    With emo_path (finishedEmoResponses.csv), distracted ratings are dropped as in the official tabulation, so
    the votes equal processedResults/tabulatedVotes.csv. A rating with a missing or malformed respLevel keeps
    its emotion vote with level None (3 rows)."""
    skip = distracted(emo_path) if emo_path else set()
    out = defaultdict(lambda: {m: {"votes": dict.fromkeys(EMOTIONS, 0), "ratings": []} for m in MODALITIES.values()})
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r["sessionNums"], r["queryType"], r["questNum"]) in skip:
                continue
            cell = out[r["clipName"]][MODALITIES[r["queryType"]]]
            cell["votes"][r["respEmo"]] += 1
            level = float(r["respLevel"]) if r["respLevel"].replace(".", "", 1).isdigit() else None
            cell["ratings"].append([r["respEmo"], level])
    return dict(out)


def vote_distribution(votes):
    n = sum(votes.values())
    return {EMOTIONS[c]: v / n for c, v in votes.items()}


def actor_folds(demographics_path, k=5, seed=0):
    """-> list of k sorted actor-ID lists. Actors are shuffled within each sex and dealt round-robin, so every
    fold gets a near-equal share of each sex; within a sex, actors are sorted by age first so the deal also
    spreads ages (the shuffle only breaks ties within the deal)."""
    with open(demographics_path, newline="") as f:
        actors = [(r["ActorID"], r["Sex"], int(r["Age"])) for r in csv.DictReader(f)]
    rng, folds, deal = random.Random(seed), [[] for _ in range(k)], 0
    for sex in sorted({s for _, s, _ in actors}):
        group = sorted((a for a in actors if a[1] == sex), key=lambda a: a[2])
        for start in range(0, len(group), k):   # shuffle within each age band of k actors, then deal
            band = group[start:start + k]
            rng.shuffle(band)
            for a in band:
                folds[deal % k].append(a[0]); deal += 1
    return [sorted(f) for f in folds]
