"""
domain.py — Zero-shot CLIP-based domain classifier.

Instead of brittle keyword matching, we embed each domain as a set of
descriptive natural-language prompts and compare incoming embeddings
against them via cosine similarity.  CLIP's shared text-image embedding
space means this works for text, image, and video inputs alike.

Domains supported:
  sports, bollywood, politics, tech, startup, food,
  travel, fitness, fashion, memes, government, general
"""

import numpy as np
import torch
import clip

# ── Re-use the already-loaded CLIP model from embeddings.py ──────────────────
# Importing model + DEVICE here avoids loading CLIP a second time in memory.
from app.ml.embeddings import model, DEVICE

# ── Domain label prompts ──────────────────────────────────────────────────────
# Each domain has multiple descriptive prompts (an "ensemble").
# CLIP averages these into ONE representative vector per domain.
# More prompts = more robust, but take slightly longer at startup.
DOMAIN_LABELS: dict[str, list[str]] = {
    "sports": [
        "a live cricket match, football game, or tennis tournament",
        "sports score, wickets, goals, or match highlights",
        "an athlete competing in a sport",
        "IPL cricket season, World Cup football, or Olympic games",
        "a player scoring runs, goals, or winning a championship",
        "stadium crowd cheering at a live sports event",
    ],
    "bollywood": [
        "a Bollywood Hindi film movie release",
        "a Hindi film song, dance, or music video",
        "an Indian movie actor starring in a film",
        "a box office movie trailer or film promotion",
        "Indian cinema award show or film festival",
    ],
    "politics": [
        "election campaign, voting, or political party rivalry",
        "a politician giving an election speech or rally",
        "political controversy, protest, or opposition debate",
        "opinion poll, election result, or political party win",
        "a political leader making policy promises",
    ],
    "government": [
        "a government official announcing a new public scheme",
        "union budget, tax reforms, or ministry circular",
        "parliament session passing a new law or bill",
        "government welfare program or infrastructure project",
        "finance minister presenting economic policy",
    ],
    "tech": [
        "a technology product or gadget",
        "software, app, or programming",
        "artificial intelligence or machine learning",
        "smartphone, laptop, or consumer electronics",
        "a technology company announcement",
    ],
    "startup": [
        "a startup company or entrepreneur",
        "venture capital funding or investment round",
        "a new business launch or pitch deck",
        "Series A, seed funding, or angel investor",
        "startup ecosystem and founder story",
    ],
    "food": [
        "a delicious meal or recipe",
        "a restaurant or café",
        "cooking food or a dish",
        "street food or cuisine",
        "food photography or a food blog post",
    ],
    "travel": [
        "a travel vlog or tourism destination guide",
        "travelling to a new country or city for vacation",
        "a solo backpacking trip or holiday adventure",
        "airport, flight boarding, or road trip journey",
        "sightseeing, hiking, or exploring tourist attractions",
        "travel photography of scenic landscapes and nature",
    ],
    "fitness": [
        "a gym workout or exercise routine",
        "yoga, running, or physical fitness",
        "a healthy lifestyle or diet",
        "bodybuilding, weight loss, or training",
        "a fitness influencer or health tip",
    ],
    "fashion": [
        "clothing, outfit, or fashion style",
        "a fashion show or runway",
        "a designer brand or clothing collection",
        "fashion trends or street style",
        "a model wearing stylish clothes",
    ],
    "memes": [
        "a funny internet meme or joke",
        "a humorous viral post",
        "comedy, satire, or parody content",
        "a relatable funny situation",
        "trending meme or internet humor",
    ],
    "general": [
        "a miscellaneous or unrelated topic",
        "everyday life or random content",
    ],
}

# Minimum confidence score to trust a prediction (0 to 1).
# If the best match is below this, we return "general".
CONFIDENCE_THRESHOLD = 0.22

# ── Keyword booster ───────────────────────────────────────────────────────────
# CLIP occasionally blurs tightly-related domains (e.g. Indian celebrity names
# score high for both sports AND bollywood because both are linked to "India"
# in CLIP's training data). This table adds a small fixed delta (+0.05) to the
# correct domain's score when known disambiguating keywords appear in the text.
# It only kicks in when scores are very close — CLIP does the heavy lifting
# for everything else.
KEYWORD_BOOSTS: dict[str, list[str]] = {
    "sports": [
        # Cricket
        "ipl", "cricket", "wicket", "century", "odi", "test match",
        "kohli", "rohit sharma", "dhoni", "bumrah", "shami",
        # Football
        "goal", "fifa", "premier league", "penalty",
        # General sport terms
        "match", "tournament", "stadium", "scorecard", "innings",
        "athlete", "medal", "olympic", "championship", "league",
    ],
    "bollywood": [
        "film", "movie", "director", "actor", "actress", "cinema",
        "bollywood", "trailer", "release date", "box office",
        "award", "filmfare", "ott", "netflix series", "web series",
    ],
    "politics": [
        "election", "voting", "ballot", "manifesto", "candidate",
        "constituency", "mp", "mla", "opposition", "ruling party",
    ],
    "government": [
        "budget", "ministry", "minister", "scheme", "policy",
        "parliament", "lok sabha", "rajya sabha", "ordinance",
    ],
    "food": [
        "recipe", "restaurant", "cuisine", "dish", "cooking",
        "chef", "meal", "ingredient", "dinner", "breakfast",
    ],
    "travel": [
        "travel", "trip", "vacation", "flight", "airport",
        "hotel", "tourism", "backpack", "itinerary", "visa",
    ],
    "fitness": [
        "workout", "gym", "exercise", "yoga", "diet",
        "weight loss", "training", "calories", "muscle", "cardio",
    ],
    "fashion": [
        "outfit", "fashion", "designer", "collection", "style",
        "brand", "clothing", "wardrobe", "trend", "runway",
    ],
    "memes": [
        "meme", "funny", "lol", "rofl", "viral", "humor",
        "relatable", "comedy", "joke", "parody",
    ],
    "startup": [
        "startup", "funding", "series a", "series b", "seed round",
        "venture capital", "angel investor", "founder", "fintech",
        "raises", "investment round", "pitch",
    ],
    "tech": [
        "iphone", "android", "ai", "software", "app",
        "coding", "gpu", "chatgpt", "algorithm", "gadget",
    ],
}
KEYWORD_BOOST_DELTA = 0.05   # how much to add to the matching domain's score


def _build_domain_embeddings() -> dict[str, np.ndarray]:
    """
    Pre-compute one representative CLIP embedding per domain at startup.

    For each domain we:
      1. Tokenise every prompt string
      2. Encode them all in one batched CLIP forward pass
      3. L2-normalise each vector
      4. Average them → one centroid per domain
      5. Re-normalise the centroid

    This runs ONCE when the module is first imported.
    """
    domain_vectors: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for domain, prompts in DOMAIN_LABELS.items():
            # Tokenise all prompts for this domain in one go
            tokens = clip.tokenize(prompts).to(DEVICE)           # (N, 77)
            embeddings = model.encode_text(tokens)                # (N, 512)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalise

            # Average over all prompts → single representative vector
            centroid = embeddings.mean(dim=0)                     # (512,)
            centroid = centroid / centroid.norm()                 # re-normalise

            domain_vectors[domain] = centroid.cpu().numpy()

    return domain_vectors


# ── Build once at import time ─────────────────────────────────────────────────
_DOMAIN_EMBEDDINGS: dict[str, np.ndarray] = _build_domain_embeddings()


# ── Public API ────────────────────────────────────────────────────────────────

def infer_domain(text: str) -> str:
    """
    Classify a text string into a domain using zero-shot CLIP.

    Steps:
      1. Embed the text via CLIP (L2-normalised 512-d vector)
      2. Compute dot product with each pre-computed domain centroid
         (equivalent to cosine similarity because both sides are normalised)
      3. Return the domain with the highest score, IF it's above
         CONFIDENCE_THRESHOLD; otherwise return "general"

    Args:
        text: The raw post caption / title / description.

    Returns:
        Domain name string, e.g. "sports", "bollywood", "politics", etc.
    """
    with torch.no_grad():
        tokens = clip.tokenize([text], truncate=True).to(DEVICE)
        text_vec = model.encode_text(tokens)                      # (1, 512)
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
        text_vec = text_vec.cpu().numpy()[0]                      # (512,)

    return infer_domain_from_embedding(text_vec, text)


def infer_domain_from_embedding(embedding: np.ndarray, text: str = "") -> str:
    """
    Classify a pre-computed CLIP embedding into a domain.

    Use this variant inside clustering.py to avoid re-embedding
    content that has already been embedded.

    Args:
        embedding: L2-normalised (512,) CLIP embedding.
        text:      Optional original text — used to apply keyword boosting
                   for CLIP's known ambiguous cases (e.g. Indian sport names
                   that bleed into the bollywood cluster).

    Returns:
        Domain name string.
    """
    # Step 1: base CLIP cosine scores
    scores: dict[str, float] = {
        domain: float(np.dot(embedding, centroid))
        for domain, centroid in _DOMAIN_EMBEDDINGS.items()
    }

    # Step 2: keyword boost — add a small delta when unambiguous keywords
    # appear in the text, to break ties that CLIP can't resolve alone
    if text:
        text_lower = text.lower()
        for domain, keywords in KEYWORD_BOOSTS.items():
            if any(kw in text_lower for kw in keywords):
                scores[domain] = scores.get(domain, 0.0) + KEYWORD_BOOST_DELTA

    # Step 3: pick the highest scoring domain
    best_domain = max(scores, key=lambda d: scores[d])
    best_score = scores[best_domain]

    # Step 4: confidence floor — if nothing matches well, return "general"
    if best_score < CONFIDENCE_THRESHOLD:
        return "general"

    return best_domain


def get_domain_scores(embedding: np.ndarray) -> dict[str, float]:
    """
    Return cosine similarity scores for ALL domains.

    Useful for debugging, logging, or returning confidence metadata
    in the API response.

    Args:
        embedding: L2-normalised (512,) CLIP embedding.

    Returns:
        Dict of {domain_name: similarity_score}, sorted by score descending.
    """
    scores = {
        domain: float(np.dot(embedding, centroid))
        for domain, centroid in _DOMAIN_EMBEDDINGS.items()
    }
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
