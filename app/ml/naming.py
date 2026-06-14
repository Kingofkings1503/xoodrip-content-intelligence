from sklearn.feature_extraction.text import TfidfVectorizer

def generate_category_name(texts, top_k=3):
    """
    Generate a human-readable category name from a list of texts
    """
    if len(texts) == 0:
        return "Miscellaneous"

    vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50,
    ngram_range=(1, 2)
    )


    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.sum(axis=0).A1
    top_indices = scores.argsort()[-top_k:][::-1]

    keywords = [feature_names[i] for i in top_indices]

    return " & ".join(word.capitalize() for word in keywords)
