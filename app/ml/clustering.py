"""
clustering.py
-------------
This is the brain of the categorization system (CategoryManager).

WHAT CHANGED vs. the old version:
  - Before: Used SQLAlchemy Session with synchronous queries
  - Now:    Uses Motor (async MongoDB driver) with await-based queries

MongoDB QUERY CHEAT SHEET (for Yash):
-------------------------------------
  SQL                              →  MongoDB (Motor)
  ─────────────────────────────────────────────────────────
  SELECT * FROM categories         →  await db.categories.find().to_list(None)
  SELECT * FROM categories         →  await db.categories.find(
    WHERE domain = 'sports'              {"domain": "sports"}
                                       ).to_list(None)
  INSERT INTO categories (...)     →  await db.categories.insert_one({...})
  UPDATE categories SET ...        →  await db.categories.update_one(
    WHERE id = 5                         {"_id": doc_id},
                                         {"$set": {"count": 6, ...}}
                                       )

  `to_list(None)` means "give me ALL matching documents as a Python list"
  `$set` means "update only these specific fields, leave the rest alone"
"""

import numpy as np
from app.ml.similarity import cosine_sim
from app.ml.domain import infer_domain_from_embedding
from app.ml.naming import generate_category_name
from app.db.model import make_category_doc, doc_to_centroid


class CategoryManager:
    """
    Manages dynamic categories using online clustering with
    domain-aware semantic gating.

    Now backed by MongoDB instead of SQLAlchemy + SQLite.
    """

    def __init__(self, db, similarity_threshold: float = 0.78):
        # `db` is a Motor database handle (e.g. db = client["xoodrip_intelligence"])
        # `db.categories` accesses the "categories" collection inside that database
        self.db = db
        self.similarity_threshold = similarity_threshold

    async def assign_category(self, embedding: np.ndarray, text: str):
        """
        Given an embedding vector and its source text, find the best-matching
        category in MongoDB, or create a new one.

        Returns a dict with the category info.
        """
        # --- Step 1: Load ALL existing categories from MongoDB ---
        # In the old code:  all_categories = self.db.query(Category).all()
        # In MongoDB:       find() with no filter = "give me everything"
        #                   to_list(None) = "collect all results into a list"
        all_categories = await self.db.categories.find().to_list(None)

        best_similarity = 0.0
        best_category = None

        # Infer the domain of the incoming content (sports, tech, etc.)
        incoming_domain = infer_domain_from_embedding(embedding, text)

        # --- Step 2: Compare incoming embedding to every stored centroid ---
        for cat_doc in all_categories:
            centroid = doc_to_centroid(cat_doc)      # list → NumPy array
            sim = cosine_sim(embedding, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_category = cat_doc

        # --- Step 3: Domain-aware guard ---
        # Even if similarity is high, don't merge across domains
        if best_category and best_category["domain"] != incoming_domain:
            best_similarity = 0.0   # force creation of a new category

        # --- Step 4: Dynamic threshold ---
        dynamic_threshold = self.similarity_threshold

        if incoming_domain == "sports":
            dynamic_threshold -= 0.08   # sports posts are more loosely grouped

        if best_category and best_category["count"] >= 3:
            dynamic_threshold += 0.03   # mature categories are stricter

        MIN_SIMILARITY = 0.65

        # --- Step 5: Decision — update existing or create new ---
        if (
            best_similarity >= dynamic_threshold
            and best_similarity >= MIN_SIMILARITY
        ) or (
            incoming_domain == "sports" and best_similarity >= 0.68
        ):
            # MATCH FOUND — update the existing category in MongoDB
            await self._update_category(best_category, embedding, text)

            return {
                "category_id": str(best_category["_id"]),   # ObjectId → string
                "is_new": False,
                "similarity": float(best_similarity),
                "domain": best_category["domain"],
                "name": best_category.get("name"),
            }
        else:
            # NO MATCH — create a brand-new category document in MongoDB
            new_doc = await self._create_category(embedding, text)

            return {
                "category_id": str(new_doc["_id"]),
                "is_new": True,
                "similarity": float(best_similarity),
                "domain": new_doc["domain"],
                "name": new_doc.get("name"),
            }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _create_category(self, embedding: np.ndarray, text: str) -> dict:
        """
        Insert a new category document into MongoDB and return it.

        Old code:  db.add(category); db.commit(); db.refresh(category)
        New code:  await db.categories.insert_one(doc)
        """
        domain = infer_domain_from_embedding(embedding, text)
        doc = make_category_doc(domain, embedding, text)

        # insert_one() adds the document to the "categories" collection
        # MongoDB automatically creates the collection if it doesn't exist!
        # After insert, `result.inserted_id` gives us the auto-generated _id
        result = await self.db.categories.insert_one(doc)
        doc["_id"] = result.inserted_id

        # If we somehow already have 3+ texts, generate a name
        if doc["count"] >= 3:
            doc["name"] = generate_category_name(doc["texts"])
            await self.db.categories.update_one(
                {"_id": doc["_id"]},
                {"$set": {"name": doc["name"]}}
            )

        return doc

    async def _update_category(self, cat_doc: dict, embedding: np.ndarray, text: str):
        """
        Update a category's centroid, count, and texts in MongoDB.

        Old code:  category.count += 1; db.commit()
        New code:  await db.categories.update_one({"_id": ...}, {"$set": {...}})
        """
        n = cat_doc["count"]
        old_centroid = doc_to_centroid(cat_doc)

        # Incremental mean: blend the old centroid with the new embedding
        new_centroid = (old_centroid * n + embedding) / (n + 1)

        new_count = n + 1
        texts = cat_doc.get("texts", [])
        texts.append(text)

        # Build the update — only change the fields that need changing
        update_fields = {
            "centroid": new_centroid.tolist(),   # NumPy → Python list
            "count": new_count,
            "texts": texts,
        }

        # Regenerate a human-readable name once we have enough examples
        if new_count >= 3:
            update_fields["name"] = generate_category_name(texts)

        # $set means "update ONLY these fields, don't touch the rest"
        await self.db.categories.update_one(
            {"_id": cat_doc["_id"]},            # filter: which document to update
            {"$set": update_fields}              # update: what to change
        )

        # Update the local dict too (so the caller gets fresh data)
        cat_doc.update(update_fields)
