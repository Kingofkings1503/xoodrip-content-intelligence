"""
clustering.py
-------------
This is the brain of the categorization system (CategoryManager).

WHAT CHANGED vs. the old version:
  - Before: categories lived in a Python list (`self.categories = []`) —
    wiped clean every time the server restarted.
  - Now:    categories live in a SQLite database file (`xoodrip.db`) —
    they survive restarts, deployments, and crashes.

The CategoryManager now receives a SQLAlchemy `db` session (a temporary
connection to the database) and uses it to:
  1. LOAD existing categories from the DB at the start of each request
  2. SAVE new/updated categories back to the DB after each decision
"""

import numpy as np
from sqlalchemy.orm import Session

from app.ml.similarity import cosine_sim
from app.ml.domain import infer_domain_from_embedding
from app.ml.naming import generate_category_name
from app.db.model import Category


class CategoryManager:
    """
    Manages dynamic categories using online clustering with
    domain-aware semantic gating.

    Now backed by a database instead of in-memory lists.
    """

    def __init__(self, db: Session, similarity_threshold: float = 0.78):
        # The database session — our open "channel" to read/write the DB
        self.db = db
        self.similarity_threshold = similarity_threshold

    def assign_category(self, embedding: np.ndarray, text: str):
        """
        Given an embedding vector and its source text, find the best-matching
        category in the database, or create a new one.

        Returns a dict with the category info.
        """
        # --- Step 1: Load ALL existing categories from the database ---
        # This replaces the old `self.categories` list that was in-memory.
        all_categories = self.db.query(Category).all()

        best_similarity = 0.0
        best_category = None

        # Infer the domain of the incoming content (sports, tech, etc.)
        # Use infer_domain_from_embedding so we don't re-embed the text through CLIP
        # (the embedding was already computed by the caller in analyze.py)
        # We also pass text so the keyword booster can activate for ambiguous cases.
        incoming_domain = infer_domain_from_embedding(embedding, text)

        # --- Step 2: Compare incoming embedding to every stored centroid ---
        for category in all_categories:
            centroid = category.get_centroid()  # deserialize bytes → NumPy array
            sim = cosine_sim(embedding, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_category = category

        # --- Step 3: Domain-aware guard ---
        # Even if similarity is high, don't merge across domains
        # (e.g. a cricket post should never go into a programming category)
        if best_category and best_category.domain != incoming_domain:
            best_similarity = 0.0  # force creation of a new category

        # --- Step 4: Dynamic threshold ---
        dynamic_threshold = self.similarity_threshold

        if incoming_domain == "sports":
            dynamic_threshold -= 0.08  # sports posts are more loosely grouped

        if best_category and best_category.count >= 3:
            dynamic_threshold += 0.03  # mature categories are stricter

        MIN_SIMILARITY = 0.65

        # --- Step 5: Decision — update existing or create new ---
        if (
            best_similarity >= dynamic_threshold
            and best_similarity >= MIN_SIMILARITY
        ) or (
            incoming_domain == "sports" and best_similarity >= 0.68
        ):
            # MATCH FOUND — update the existing category in the DB
            self._update_category(best_category, embedding, text)

            return {
                "category_id": best_category.id,
                "is_new": False,
                "similarity": float(best_similarity),
                "domain": best_category.domain,
                "name": best_category.name,
            }
        else:
            # NO MATCH — create a brand-new category row in the DB
            new_category = self._create_category(embedding, text)

            return {
                "category_id": new_category.id,
                "is_new": True,
                "similarity": float(best_similarity),
                "domain": new_category.domain,
                "name": new_category.name,
            }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _create_category(self, embedding: np.ndarray, text: str) -> Category:
        """
        Insert a new Category row into the database and return it.
        """
        category = Category(
            domain=infer_domain_from_embedding(embedding, text),
            count=1,
        )
        category.set_centroid(embedding)   # serialize NumPy → bytes
        category.set_texts([text])         # serialize list → JSON string

        self.db.add(category)
        self.db.commit()
        self.db.refresh(category)  # loads the auto-assigned `id` back into the object

        # After commit, try to set a name if we somehow already have 3+ texts
        if category.count >= 3:
            category.name = generate_category_name(category.get_texts())
            self.db.commit()

        return category

    def _update_category(self, category: Category, embedding: np.ndarray, text: str):
        """
        Update a category's centroid, count, and texts, then save to the DB.
        """
        n = category.count
        old_centroid = category.get_centroid()

        # Incremental mean: blend the old centroid with the new embedding
        new_centroid = (old_centroid * n + embedding) / (n + 1)

        category.set_centroid(new_centroid)
        category.count += 1

        texts = category.get_texts()
        texts.append(text)
        category.set_texts(texts)

        # Regenerate a human-readable name once we have enough examples
        if category.count >= 3:
            category.name = generate_category_name(category.get_texts())

        # Save everything to the database
        self.db.commit()
        self.db.refresh(category)
