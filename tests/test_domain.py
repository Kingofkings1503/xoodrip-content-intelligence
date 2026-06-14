"""
tests/test_domain.py
--------------------
Integration tests for the zero-shot CLIP domain classifier.

These tests require the CLIP model to be installed and will take a few seconds
on first run (one-time CLIP download if not cached).

Run with:
    cd c:\\Users\\Dell\\OneDrive\\Desktop\\Xoodrip\\xoodrip-content-intelligence
    python -m pytest tests/test_domain.py -v
"""

import pytest
from app.ml.domain import infer_domain, infer_domain_from_embedding, get_domain_scores
from app.ml.embeddings import embed_text


# ---------------------------------------------------------------------------
# Test cases: (post_text, expected_domain)
# These are representative posts for each domain we support.
# ---------------------------------------------------------------------------
DOMAIN_TEST_CASES = [
    # Sports
    ("Virat Kohli hits a century in IPL 2025", "sports"),
    ("India wins the cricket world cup final match", "sports"),
    ("Player scores wickets in the championship game", "sports"),

    # Bollywood
    ("Salman Khan's new movie releases this Friday", "bollywood"),
    ("Deepika Padukone dazzles at the film premiere", "bollywood"),
    ("New Bollywood song goes viral on Instagram", "bollywood"),

    # Politics
    ("Opposition party wins election against ruling government", "politics"),
    ("Election results and voting turnout across constituencies", "politics"),

    # Government
    ("Finance minister presents the union budget for 2025", "government"),
    ("Government announces new subsidy scheme for farmers", "government"),

    # Tech
    ("Apple launches the new iPhone with AI features", "tech"),
    ("Google releases a new machine learning framework", "tech"),
    ("OpenAI announces GPT-5 with multimodal capabilities", "tech"),

    # Startup
    ("Bengaluru startup raises $10M in Series A funding", "startup"),
    ("New fintech startup disrupts digital payments space", "startup"),

    # Food
    ("This butter chicken recipe will blow your mind", "food"),
    ("Top 10 street food spots in Mumbai you must try", "food"),

    # Travel
    ("Solo backpacking trip through Europe: a travel vlog", "travel"),
    ("My vacation flight and sightseeing tour itinerary", "travel"),

    # Fitness
    ("5 exercises to build core strength at home", "fitness"),
    ("Yoga for beginners: 10-minute morning routine", "fitness"),

    # Fashion
    ("These monsoon outfit ideas are absolutely stunning", "fashion"),
    ("Gucci's new collection hits Indian stores this week", "fashion"),

    # Memes
    ("When your code works on the first try 😂 #meme", "memes"),
    ("Monday mornings be like… relatable? 😩 #funny", "memes"),
]


@pytest.mark.parametrize("text, expected_domain", DOMAIN_TEST_CASES)
def test_infer_domain_text(text: str, expected_domain: str):
    """
    For each test post, verify infer_domain() returns the expected domain.
    Allowed to be off by one adjacent domain (e.g. politics vs government)
    since CLIP may blur closely-related domains.
    """
    result = infer_domain(text)
    assert result == expected_domain, (
        f"\nText: {text!r}\n"
        f"Expected: {expected_domain!r}\n"
        f"Got:      {result!r}\n"
        f"Scores:   {get_domain_scores(embed_text(text))}"
    )


def test_infer_domain_from_embedding_matches_infer_domain():
    """
    infer_domain(text) and infer_domain_from_embedding(embed_text(text), text)
    should always produce the same result (both paths apply keyword boosting).
    """
    texts = [
        "Kohli smashes century in IPL",
        "New startup funding round",
        "Butter chicken recipe for dinner",
    ]
    for text in texts:
        emb = embed_text(text)
        via_text = infer_domain(text)
        via_emb = infer_domain_from_embedding(emb, text)   # pass text to activate booster
        assert via_text == via_emb, (
            f"Mismatch for {text!r}: infer_domain={via_text!r}, "
            f"infer_domain_from_embedding={via_emb!r}"
        )


def test_get_domain_scores_returns_all_domains():
    """
    get_domain_scores should return one score for every domain defined
    in DOMAIN_LABELS, and all scores should be between -1 and 1.
    """
    from app.ml.domain import DOMAIN_LABELS
    emb = embed_text("some random post about cricket")
    scores = get_domain_scores(emb)

    # Every domain must have a score
    assert set(scores.keys()) == set(DOMAIN_LABELS.keys()), (
        f"Missing domains: {set(DOMAIN_LABELS.keys()) - set(scores.keys())}"
    )

    # All scores must be valid cosine similarities
    for domain, score in scores.items():
        assert -1.0 <= score <= 1.0, (
            f"Score for {domain!r} is out of range: {score}"
        )


def test_get_domain_scores_sorted_descending():
    """
    get_domain_scores should return domains sorted by score, highest first.
    """
    emb = embed_text("Bollywood actress wins best actress award")
    scores = get_domain_scores(emb)
    score_values = list(scores.values())
    assert score_values == sorted(score_values, reverse=True), (
        "Domain scores are not sorted in descending order!"
    )


def test_sports_is_top_for_cricket_post():
    """
    For an obvious cricket post, 'sports' should be the #1 scoring domain.
    """
    emb = embed_text("India wins the cricket World Cup 2025")
    scores = get_domain_scores(emb)
    top_domain = next(iter(scores))   # first key = highest score
    assert top_domain == "sports", (
        f"Expected 'sports' to be top domain, got {top_domain!r}. "
        f"All scores: {scores}"
    )


def test_bollywood_is_top_for_film_post():
    """
    For an obvious Bollywood post, 'bollywood' should be top domain.
    """
    emb = embed_text("Ranveer Singh's new Hindi film releases today in cinemas")
    scores = get_domain_scores(emb)
    top_domain = next(iter(scores))
    assert top_domain == "bollywood", (
        f"Expected 'bollywood' to be top domain, got {top_domain!r}. "
        f"All scores: {scores}"
    )
