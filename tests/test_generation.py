from src.generation.llm import parse_answer
from src.generation.prompts import INSUFFICIENT_EVIDENCE, build_user_prompt, format_evidence

CHUNKS = [
    {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director."},
    {"title": "Ed Wood", "text": "Ed Wood was an American filmmaker."},
]


def test_evidence_is_numbered_from_one_with_titles():
    assert format_evidence(CHUNKS) == (
        "[1] (Scott Derrickson) Scott Derrickson is an American director.\n"
        "[2] (Ed Wood) Ed Wood was an American filmmaker."
    )


def test_user_prompt_contains_evidence_and_question():
    prompt = build_user_prompt("Same nationality?", CHUNKS)
    assert prompt.startswith("Evidence:\n[1]")
    assert prompt.endswith("Question: Same nationality?")


def test_parse_valid_json():
    assert parse_answer('{"answer": "yes", "citations": [1, 2]}', 2) == ("yes", [1, 2])


def test_parse_drops_out_of_range_and_duplicate_citations():
    assert parse_answer('{"answer": "yes", "citations": [0, 1, 1, 5, "x"]}', 2) == ("yes", [1])


def test_parse_empty_answer_becomes_insufficient():
    assert parse_answer('{"answer": "", "citations": []}', 2) == (INSUFFICIENT_EVIDENCE, [])


def test_parse_non_json_falls_back_to_raw_text():
    assert parse_answer("yes", 2) == ("yes", [])
