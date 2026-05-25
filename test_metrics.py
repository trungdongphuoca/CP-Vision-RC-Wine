import re


def parse_semantic_id(text):
    match = re.search(r"[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-(?:\d{4}|NV)", text)
    return match.group(0) if match else text.strip()[:20]


def eval_components(pred_id, target_id):
    pred_parts = pred_id.split("-")
    target_parts = target_id.split("-")

    country_match = int(pred_parts[0] == target_parts[0]) if pred_parts and target_parts else 0
    variety_match = (
        int(pred_parts[2] == target_parts[2])
        if len(pred_parts) > 2 and len(target_parts) > 2
        else 0
    )
    vintage_match = (
        int(pred_parts[3] == target_parts[3])
        if len(pred_parts) > 3 and len(target_parts) > 3
        else 0
    )
    intent_match = int(country_match and variety_match)

    return country_match, variety_match, vintage_match, intent_match


def test_parse_semantic_id_extracts_id_from_generated_text():
    text = "I suggest [US-CALI-CABE-2015]. A bold wine."

    assert parse_semantic_id(text) == "US-CALI-CABE-2015"


def test_parse_semantic_id_falls_back_to_trimmed_text():
    assert parse_semantic_id("no id here") == "no id here"


def test_eval_components_separates_exact_region_from_intent():
    pred = "US-CALI-CABE-2015"
    target = "US-NEWY-CABE-2015"

    assert eval_components(pred, target) == (1, 1, 1, 1)


def test_eval_components_handles_malformed_prediction():
    assert eval_components("bad", "US-CALI-CABE-2015") == (0, 0, 0, 0)
