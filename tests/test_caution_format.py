from engine.scan import mark_caution


def test_marks_only_the_word_and_keeps_case():
    assert mark_caution("News check: caution · Analyst downgrade") == "News check: 🔴 **caution** · Analyst downgrade"
    assert mark_caution("- XYZ: CAUTION. News: guidance cut.") == "- XYZ: 🔴 **CAUTION**. News: guidance cut."


def test_idempotent_and_whole_words_only():
    s = "REVIEW ABC position (caution): sector selloff; precautionary note"
    once = mark_caution(s)
    assert once == "REVIEW ABC position (🔴 **caution**): sector selloff; precautionary note"
    assert mark_caution(once) == once
    assert mark_caution("No news issues.") == "No news issues."
