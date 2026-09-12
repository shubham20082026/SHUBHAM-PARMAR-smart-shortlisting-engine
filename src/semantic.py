"""
Semantic (meaning-based) matching between JD requirement chunks and resume
chunks.

Primary approach: sentence-transformers embeddings ("all-MiniLM-L6-v2"),
compared with cosine similarity. This is what you tell judges you're using
for the SEMANTIC arm -> it's what lets "built REST APIs with Express and
MongoDB" match a JD line about "Node.js backend development" even with
zero literal keyword overlap.

Fallback: if sentence-transformers isn't installed / no internet at venue
to download the model, we automatically fall back to TF-IDF + cosine
similarity (still "semantic-ish" via shared vocabulary weighting, weaker
than embeddings but keeps the demo alive). The fallback is logged loudly
so you know which mode you're in before you present.
"""

from typing import List, Tuple
import numpy as np

_MODE = None
_model = None


def _load_model():
    global _MODE, _model
    if _MODE is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _MODE = "embeddings"
        print("[semantic] Using sentence-transformers (all-MiniLM-L6-v2)")
    except Exception as e:
        _MODE = "tfidf"
        print(f"[semantic] sentence-transformers unavailable ({e}); "
              f"falling back to TF-IDF cosine similarity")


def get_mode() -> str:
    _load_model()
    return _MODE


def _embed(texts: List[str]) -> np.ndarray:
    _load_model()
    if _MODE == "embeddings":
        return np.array(_model.encode(texts, show_progress_bar=False))
    else:
        # TF-IDF fallback: fit per-comparison so vocab is shared between
        # the two documents being compared.
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(texts)
        return mat.toarray()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_similarity_matrix(jd_chunks: List[str], resume_chunks: List[str]) -> np.ndarray:
    """
    Returns an (len(jd_chunks) x len(resume_chunks)) matrix of cosine
    similarities between every JD requirement chunk and every resume chunk.
    """
    all_texts = jd_chunks + resume_chunks
    if get_mode() == "tfidf":
        # must embed jointly so TF-IDF vocab is shared
        embs = _embed(all_texts)
    else:
        embs = _embed(all_texts)
    jd_embs = embs[:len(jd_chunks)]
    resume_embs = embs[len(jd_chunks):]
    sim = np.zeros((len(jd_chunks), len(resume_chunks)))
    for i, je in enumerate(jd_embs):
        for j, re_ in enumerate(resume_embs):
            sim[i, j] = _cosine(je, re_)
    return sim


def semantic_score(jd_chunks: List[str], resume_chunks: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Computes an overall semantic fit score in [0,1]:
    for each JD requirement chunk, take its BEST matching resume chunk
    (max similarity), then average across all JD chunks. This rewards
    resumes that cover every JD requirement somewhere, rather than being
    diluted by irrelevant resume content (education, hobbies, etc.)

    Returns (score, best_matches) where best_matches is a list of
    (jd_chunk, best_resume_chunk, similarity) for inspection/explanation.
    """
    if not jd_chunks or not resume_chunks:
        return 0.0, []
    sim = chunk_similarity_matrix(jd_chunks, resume_chunks)
    best_matches = []
    per_requirement_best = []
    for i, jd_chunk in enumerate(jd_chunks):
        best_j = int(np.argmax(sim[i]))
        best_sim = float(sim[i, best_j])
        per_requirement_best.append(best_sim)
        best_matches.append((jd_chunk, resume_chunks[best_j], best_sim))
    score = float(np.mean(per_requirement_best))
    return score, best_matches
