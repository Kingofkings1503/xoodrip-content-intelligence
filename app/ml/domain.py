"""
domain.py — Zero-shot SigLIP-based domain classifier.

WHAT CHANGED vs. the old CLIP version:
  - Replaced `clip.tokenize()` with `processor(text=...)`
  - Replaced `model.encode_text()` with `model.get_text_features()`
  - Embedding dimension: 512 → 1152
  - Confidence threshold recalibrated for SigLIP's score distribution
  - Same domain labels, keyword boosts, and public API

Instead of brittle keyword matching, we embed each domain as a set of
descriptive natural-language prompts and compare incoming embeddings
against them via cosine similarity.  SigLIP's shared text-image embedding
space means this works for text, image, and video inputs alike.

Domains supported:
  sports, bollywood, politics, tech, startup, food,
  travel, fitness, fashion, memes, government, general
"""
import numpy as np
import torch
import torch.nn.functional as F

# ── Re-use the already-loaded SigLIP model from embeddings.py ────────────────
# Importing model, processor, and DEVICE here avoids loading SigLIP a second
# time in memory.
from app.ml.embeddings import model, processor, DEVICE

# ── Domain label prompts ──────────────────────────────────────────────────────
# Each domain has multiple descriptive prompts (an "ensemble").
# SigLIP averages these into ONE representative vector per domain.
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

# ── Confidence threshold ──────────────────────────────────────────────────────
# SigLIP's sigmoid-based scores have a different magnitude than CLIP's softmax.
# SigLIP scores tend to be slightly lower overall but with better separation
# between correct and incorrect domains. We lower the threshold to 0.18
# (from CLIP's 0.22) to account for this.
CONFIDENCE_THRESHOLD = 0.18

# ── Keyword booster ───────────────────────────────────────────────────────────
# SigLIP is better at domain separation than CLIP, but keyword boosting
# still helps for closely-related Indian domains where training data overlap.
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
    Pre-compute one representative SigLIP embedding per domain at startup.

    For each domain we:
      1. Tokenise every prompt string via the SigLIP processor
      2. Encode them all via model.get_text_features()
      3. L2-normalise each vector
      4. Average them → one centroid per domain
      5. Re-normalise the centroid

    Old CLIP way:
        tokens = clip.tokenize(prompts).to(DEVICE)
        embeddings = model.encode_text(tokens)

    New SigLIP way:
        inputs = processor(text=prompts, ...)
        embeddings = model.get_text_features(**inputs)

    This runs ONCE when the module is first imported.
    """
    domain_vectors: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for domain, prompts in DOMAIN_LABELS.items():
            # Tokenise all prompts for this domain in one batch
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Encode via SigLIP                              # (N, 1152)
            kwargs = {
                "input_ids": inputs["input_ids"],
            }

            if "attention_mask" in inputs:
                kwargs["attention_mask"] = inputs["attention_mask"]

            embeddings = model.get_text_features(**kwargs)
            if hasattr(embeddings, "pooler_output"):
                embeddings = embeddings.pooler_output
            elif not isinstance(embeddings, torch.Tensor) and hasattr(embeddings, "last_hidden_state"):
                embeddings = embeddings.last_hidden_state.mean(dim=1)
                
            embeddings = F.normalize(embeddings, p=2, dim=-1)  # L2 normalise

            # Average over all prompts → single representative vector
            centroid = embeddings.mean(dim=0)                  # (1152,)
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=-1)[0]

            domain_vectors[domain] = centroid.cpu().numpy()

    return domain_vectors


# ── Build once at import time ─────────────────────────────────────────────────
print("[INFO] Building domain embeddings...")
_DOMAIN_EMBEDDINGS: dict[str, np.ndarray] = _build_domain_embeddings()
print(f"[SUCCESS] Domain embeddings built for {len(_DOMAIN_EMBEDDINGS)} domains")


# ── Public API ────────────────────────────────────────────────────────────────

def infer_domain(text: str) -> str:
    """
    Classify a text string into a domain using zero-shot SigLIP.

    Steps:
      1. Embed the text via SigLIP (L2-normalised 1152-d vector)
      2. Compute dot product with each pre-computed domain centroid
         (equivalent to cosine similarity because both sides are normalised)
      3. Return the domain with the highest score, IF it's above
         CONFIDENCE_THRESHOLD; otherwise return "general"

    Args:
        text: The raw post caption / title / description.

    Returns:
        Domain name string, e.g. "sports", "bollywood", "politics", etc.
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        text_vec = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_vec = F.normalize(text_vec, p=2, dim=-1)
        text_vec = text_vec.cpu().numpy()[0]                  # (1152,)

    return infer_domain_from_embedding(text_vec, text)


def infer_domain_from_embedding(embedding: np.ndarray, text: str = "") -> str:
    """
    Classify a pre-computed SigLIP embedding into a domain.

    Use this variant inside clustering.py to avoid re-embedding
    content that has already been embedded.

    Args:
        embedding: L2-normalised (1152,) SigLIP embedding.
        text:      Optional original text — used to apply keyword boosting
                   for known ambiguous cases.

    Returns:
        Domain name string.
    """
    # Step 1: base SigLIP cosine scores
    scores: dict[str, float] = {
        domain: float(np.dot(embedding, centroid))
        for domain, centroid in _DOMAIN_EMBEDDINGS.items()
    }

    # Step 2: keyword boost — add a small delta when unambiguous keywords
    # appear in the text, to break ties that SigLIP can't resolve alone
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
        embedding: L2-normalised (1152,) SigLIP embedding.

    Returns:
        Dict of {domain_name: similarity_score}, sorted by score descending.
    """
    scores = {
        domain: float(np.dot(embedding, centroid))
        for domain, centroid in _DOMAIN_EMBEDDINGS.items()
    }
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
