from app.ml.embeddings import embed_text
from app.ml.similarity import cosine_sim

vec1 = embed_text("Budget announced by Indian government")
vec2 = embed_text("Finance minister presents annual budget")
vec3 = embed_text("Virat Kohli hits a century in IPL match")

print("Budget vs Finance:", cosine_sim(vec1, vec2))
print("Budget vs Cricket:", cosine_sim(vec1, vec3))
