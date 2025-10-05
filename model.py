from sentence_transformers import SentenceTransformer, models

def get_model(model_name: str, pooling_mode: str = 'mean'):
    word_emb = models.Transformer(model_name)
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling_mode == 'mean'),
        pooling_mode_cls_token=(pooling_mode == 'cls'),
        pooling_mode_max_tokens=(pooling_mode == 'max')
    )
    model = SentenceTransformer(modules=[word_emb, pooling])
    return model
