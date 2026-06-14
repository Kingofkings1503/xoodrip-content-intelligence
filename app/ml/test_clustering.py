from app.ml.embeddings import embed_text
from app.ml.clustering import CategoryManager

manager = CategoryManager(similarity_threshold=0.78)

posts = [
    "Budget announced by Indian government",
    "Finance minister presents annual budget",
    "Virat Kohli hits a century in IPL",
    "Rohit Sharma scores a double hundred",
    "New startup funding announced in Bengaluru"
]

for post in posts:
    emb = embed_text(post)
    result = manager.assign_category(emb,post)
    print(post)
    print(result)
    print("-" * 50)
