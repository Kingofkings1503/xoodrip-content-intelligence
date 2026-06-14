from app.ml.embeddings import embed_text, embed_image

text_vec = embed_text("Budget announced by Indian government")
print("Text embedding shape:", text_vec.shape)

# Put any image path here
# img_vec = embed_image("sample.jpg")
# print("Image embedding shape:", img_vec.shape)
