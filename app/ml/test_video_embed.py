from app.ml.embeddings import embed_video

video_path = r"C:\Users\Dell\OneDrive\Desktop\Xoodrip\xoodrip-content-intelligence\app\ml\sample.mp4"  # put any short video here

vec = embed_video(video_path)

print("Video embedding shape:", vec.shape)
print("Norm:", (vec**2).sum() ** 0.5)
