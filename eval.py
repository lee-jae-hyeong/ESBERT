from sentence_transformers import util

def evaluate(model, eval_csv):
    import pandas as pd
    df = pd.read_csv(eval_csv)

    emb1 = model.encode(df['anchor'].tolist(), convert_to_tensor=True)
    emb2 = model.encode(df['positive'].tolist(), convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb1, emb2)
    mean_score = cosine_scores.diag().mean().item()
    print(f"Mean Cosine Similarity: {mean_score:.4f}")
