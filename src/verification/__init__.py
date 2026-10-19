"""Evidence verification and confidence scoring."""

from src.verification.confidence import ConfidenceResult, DEFAULT_THRESHOLD, score_confidence
from src.verification.nli import DEFAULT_NLI, NLIVerifier, VerificationResult

__all__ = [
    "ConfidenceResult",
    "DEFAULT_NLI",
    "DEFAULT_THRESHOLD",
    "NLIVerifier",
    "VerificationResult",
    "score_confidence",
]
