from src.generation.prompts import INSUFFICIENT_EVIDENCE
from src.retrieval.search import SearchHit
from src.verification.confidence import score_confidence
from src.verification.nli import NLIVerifier, VerificationResult, build_hypothesis, evidence_text


class FakeNLI:
    """Maps hypothesis substrings to fixed logits [contradiction, entailment, neutral]."""

    def __init__(self, table: dict[str, list[float]], default: list[float] | None = None):
        self.table = table
        self.default = default or [0.0, 0.0, 3.0]
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        out = []
        for _, hyp in pairs:
            for key, logits in self.table.items():
                if key in hyp:
                    out.append(logits)
                    break
            else:
                out.append(self.default)
        return out


CHUNKS = [
    {"title": "Ortaköy", "sent_id": 0, "text": "The mansion is in Ortaköy."},
    {"title": "Laleli", "sent_id": 0, "text": "The mosque is in Laleli."},
]


def test_build_hypothesis_yes_no_and_span():
    assert build_hypothesis("Are they in the same neighborhood?", "yes") == (
        "Are they in the same neighborhood"
    )
    assert "not the case" in build_hypothesis("Are they in the same neighborhood?", "no").lower()
    assert "Tim Burton" in build_hypothesis("Who directed Ed Wood?", "Tim Burton")


def test_evidence_prefers_citations():
    text, used = evidence_text(CHUNKS, [2])
    assert used is True
    assert "Laleli" in text and "Ortaköy" not in text
    text_all, used_all = evidence_text(CHUNKS, [])
    assert used_all is False
    assert "Ortaköy" in text_all and "Laleli" in text_all


def test_yes_no_prefers_stronger_side():
    # Affirmative hypothesis is weakly entailed; negated is strongly entailed.
    scorer = FakeNLI(
        {
            "not the case": [0.1, 3.0, 0.1],  # "no" side wins
            "Are they": [0.1, 0.5, 2.0],
        }
    )
    result = NLIVerifier(model=scorer).verify(
        "Are they in the same neighborhood?", "yes", CHUNKS, [1, 2]
    )
    assert result.supported is False
    assert result.rival_entailment is not None
    assert result.rival_entailment > result.entailment


def test_span_answer_supported_when_not_clearly_contradicted():
    # Neutral-dominant should still be supported for short spans.
    scorer = FakeNLI({"answer to the question": [0.5, 0.8, 2.5]})
    result = NLIVerifier(model=scorer).verify("Where is the mosque?", "Laleli", CHUNKS, [2])
    assert result.supported is True
    assert result.neutral > result.contradiction


def test_span_answer_rejected_when_contradiction_is_strongest():
    scorer = FakeNLI({"answer to the question": [3.0, 0.2, 0.5]})
    result = NLIVerifier(model=scorer).verify("Where is the mosque?", "Paris", CHUNKS, [2])
    assert result.supported is False


def test_verify_insufficient_skips_model():
    scorer = FakeNLI({})
    result = NLIVerifier(model=scorer).verify("Q?", INSUFFICIENT_EVIDENCE, CHUNKS, [])
    assert result.supported is False
    assert scorer.pairs is None


def test_confidence_accepts_and_rejects():
    supported = VerificationResult(
        supported=True, entailment=0.8, contradiction=0.1, neutral=0.1,
        premise="p", hypothesis="h", used_citations=True,
    )
    hits = [SearchHit(score=2.0, chunk=CHUNKS[0])]
    ok = score_confidence(supported, hits, [1], threshold=0.35)
    assert ok.accept is True
    assert ok.score >= 0.35

    weak = VerificationResult(
        supported=False, entailment=0.1, contradiction=0.7, neutral=0.2,
        premise="p", hypothesis="h", used_citations=False,
    )
    bad = score_confidence(weak, hits, [], threshold=0.35)
    assert bad.accept is False
