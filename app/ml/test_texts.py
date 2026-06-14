from app.ml.embeddings import embed_text
from app.ml.clustering import CategoryManager

manager = CategoryManager(similarity_threshold=0.78)

test_posts = [
    # Sports
    "Virat Kohli hits a century in IPL",
    "Rohit Sharma scores 100 in ODI",
    "India wins cricket match against Australia",
    "IPL final ends in thrilling finish",
    "Cricket world cup match highlights",

    # Government
    "Finance minister presents union budget",
    "New tax reforms introduced by government",
    "Parliament passes new economic bill",
    "Government announces infrastructure plan",

    # Startup
    "Indian startup raises Series A funding",
    "New fintech startup launched in Bengaluru",
    "Venture capital firms invest in AI startups",
    "Startup founders pitch to investors",

    # Noise / Random
    "Weather is very hot today",
    "I love eating pizza on weekends",
    "Watching movies at night is relaxing"
]

for post in test_posts:
    emb = embed_text(post)
    result = manager.assign_category(emb, post)

    print("\nPOST:", post)
    print("RESULT:", result)
