"""
model.py
--------
Helper functions for working with Category documents in MongoDB.

WHAT CHANGED vs. the old SQLAlchemy version:
--------------------------------------------
Before (SQLAlchemy + SQLite):
  - We had a `Category` Python class with `Column(Integer, ...)` etc.
  - SQLAlchemy mapped each class attribute to a SQL column
  - We had to pickle NumPy arrays into bytes (SQL can't store arrays)
  - We had to JSON-encode lists into strings

After (MongoDB):
  - No ORM class needed! MongoDB stores JSON-like documents natively.
  - A "category" is just a Python dict:
      {
          "name": "Cricket & Ipl",
          "domain": "sports",
          "count": 5,
          "centroid": [0.012, -0.034, ...],   ← stored as a native array!
          "texts": ["Kohli hits century", "India wins match"]
      }
  - NumPy arrays → just convert to a Python list (no pickle!)
  - Python lists → stored directly (no JSON encoding!)
  - MongoDB auto-adds an `_id` field (like SQL's auto-increment primary key,
    but it's a unique string called ObjectId instead of an integer)

WHY IS THIS SIMPLER?
  - No pickle.dumps / pickle.loads
  - No json.dumps / json.loads
  - No Column definitions or table schemas
  - MongoDB figures out the "schema" from the documents you insert
"""

import numpy as np


def make_category_doc(domain: str, centroid: np.ndarray, text: str) -> dict:
    """
    Build a new category document ready to insert into MongoDB.

    Args:
        domain:   The broad topic (e.g. "sports", "tech")
        centroid: The 1152-d SigLIP embedding as a NumPy array
        text:     The first post text assigned to this category

    Returns:
        A dict that looks like a MongoDB document:
        {
            "name": None,
            "domain": "sports",
            "count": 1,
            "centroid": [0.012, -0.034, ...],
            "texts": ["Kohli hits century"]
        }
    """
    return {
        "name": None,                       # filled later when count >= 3
        "domain": domain,
        "count": 1,
        "centroid": centroid.tolist(),        # NumPy array → Python list
        "texts": [text],
    }


def doc_to_centroid(doc: dict) -> np.ndarray:
    """
    Convert a MongoDB document's centroid field back into a NumPy array.

    In the old code this was:  pickle.loads(category.centroid)
    Now it's just:             np.array(doc["centroid"])
    """
    return np.array(doc["centroid"], dtype=np.float32)
