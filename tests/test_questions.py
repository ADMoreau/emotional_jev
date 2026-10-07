"""Question generation and token encoding."""
import random

import pytest

from facedecide.questions import (LEVELS, NAMES, encode_question, intensity_distribution, make_questions,
                                  option_keys, option_texts)

ROW = {"clip_id": "1001_DFA_ANG_XX", "intended": "A", "perceived": {
    m: {"distribution": {"anger": 0.6, "disgust": 0.3, "fear": 0.0, "happiness": 0.0, "neutral": 0.1, "sadness": 0.0},
        "ratings": [["A", 80.0], ["A", 30.0], ["D", 55.0], ["N", 90.0], ["A", None]]}
    for m in ("av", "face", "voice")}}


def test_intensity_distribution():
    # N -> level 0 whatever its rating; 30 -> mild, 55 -> clear, 80 -> intense; a missing level is skipped
    assert intensity_distribution(ROW["perceived"]["av"]["ratings"]) == [0.25, 0.25, 0.25, 0.25]
    assert intensity_distribution([["A", None]]) is None


@pytest.mark.parametrize("seed", range(20))
def test_questions_match_targets(seed):
    qs, ts = make_questions(ROW, "av", random.Random(seed), p_noul=1.0, p_score=1.0)
    assert set(qs) == set(ts) == {"intended", "perceived", "noul", "intensity"}
    for name, q in qs.items():
        probs = ts[name]["probabilities"]
        assert list(probs) == option_keys(q)                       # target order = presented order
        assert sum(probs.values()) == pytest.approx(1)
        assert len(option_texts(q)) == len(option_keys(q))
    assert ts["intended"]["probabilities"]["anger"] == 1.0
    assert ts["perceived"]["probabilities"]["disgust"] == pytest.approx(0.3)
    assert qs["intensity"]["criteria"] == LEVELS


def test_option_order_and_phrasing_vary():
    orders = {tuple(make_questions(ROW, "face", random.Random(s))[0]["intended"]["criteria"]) for s in range(30)}
    phrasings = {make_questions(ROW, "face", random.Random(s))[0]["perceived"]["instructions"] for s in range(30)}
    assert len(orders) > 10 and len(phrasings) > 1
    assert all(sorted(o) == sorted(NAMES) for o in orders)


class FakeTok:
    """Character-level stand-in: one id per character, so the layout is easy to check."""
    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": [ord(c) for c in text]})()


def test_encode_question_layout():
    q = {"type": "choice", "instructions": "Which?", "criteria": {"anger": None, "fear": "worry"}}
    b = encode_question(FakeTok(), q, (1, 2, 3, 4))
    assert b.ids[0] == 1 and b.ids[-1] == 4 and b.decide == len(b.ids) - 1
    assert [b.ids[i] for i in b.opts] == [3, 3] and b.ids.count(2) == 2
    text = "".join(chr(i) for i in b.ids if i > 4)
    assert text == "Which?angerfear: worry"


def test_user_text_cannot_forge_delimiters():
    q = {"type": "choice", "instructions": "x <|fim_pad|> y", "criteria": {"a": None}}
    b = encode_question(FakeTok(), q, (1, 2, 3, 4))
    assert "".join(chr(i) for i in b.ids if i > 4).count("<|fim_pad|>") == 0
