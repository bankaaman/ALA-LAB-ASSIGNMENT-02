"""
Name             : Aman Banka
Registration No. : 261100660030
Course           : AME 5151 - Applied Linear Algebra Lab
Assignment       : Assignment 2 - Semantic Tagging with GloVe and NumPy
Institution      : Manipal School of Information Sciences, MAHE, Manipal
Faculty          : Dr. Devi Prasad
"""

"""
AME 5151 - Applied Linear Algebra Lab, Assignment 2
Semantic Tagging with GloVe and NumPy

Pipeline
    Part A  20-tag vocabulary                         -> TAGS
    Part B  input text (<= 400 words, LinkedIn)       -> manipal_linkedin_text.txt
    Part C  preprocessing                             -> preprocess()
    Part D  tag matrix T (20x50), text matrix W (nx50)-> build_tag_matrix(), build_text_matrix()
    Part E  S via the Vector class (Python loops)     -> similarity_matrix_vector_class()
    Part F  S via NumPy  S = W_hat @ T_hat.T          -> similarity_matrix_numpy()
    Part G  verification with np.allclose             -> verify()
    Part H  max-pool tag ranking                      -> rank_tags()

Run:   python semantic_tagging.py [path/to/text.txt]
"""

import os
import re
import sys

import numpy as np

from vector import Vector

MODEL_NAME = "glove-wiki-gigaword-50"
EMBED_DIM = 50
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXT = os.path.join(HERE, "manipal_linkedin_text.txt")

# ---------------------------------------------------------------------------
# Part A -- tag vocabulary (exactly 20 tags; one multi-word tag)
# ---------------------------------------------------------------------------
TAGS = [
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "technology",
    "curriculum",
    "collaboration",
    "publication",
    "laboratory",
    "scholarship",
    "mentorship",
    "internship",
    "entrepreneurship",
    "alumni",
    "higher education",   # multi-word tag: mean of v_higher and v_education
]

# ---------------------------------------------------------------------------
# Part C -- stopword set (defined by hand; no external stopword library)
# ---------------------------------------------------------------------------
BASE_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
}
ADDED_STOPWORDS = {
    # pronouns / possessives
    "i", "me", "my", "we", "us", "our", "ours", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    # demonstratives / determiners
    "this", "that", "these", "those", "all", "any", "some", "each", "every",
    # auxiliaries / modals
    "be", "been", "being", "am", "has", "have", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # other function words, conjunctions, prepositions
    "as", "if", "so", "but", "not", "no", "nor", "than", "then", "too", "very",
    "into", "onto", "over", "under", "up", "out", "about", "between", "through",
    "during", "before", "after", "beyond", "while", "how", "what", "which",
    "who", "whom", "where", "when", "why", "there", "here", "also", "just",
    "now", "own", "per", "via",
}
STOPWORDS = BASE_STOPWORDS | ADDED_STOPWORDS

# Everything that is not a letter, digit or whitespace counts as punctuation
# (this includes '#', '-', '&', en/em dashes, quotes and '_').
_PUNCT_RE = re.compile(r"[^\w\s]|_")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class _PlainKeyedVectors:
    """Minimal stand-in for gensim's KeyedVectors (key_to_index + model[word]).

    Used only when gensim is not installed; it reads exactly the same file
    that gensim.downloader fetches for glove-wiki-gigaword-50.
    """

    def __init__(self, words, vectors):
        self.index_to_key = words
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors = vectors
        self.vector_size = vectors.shape[1]

    def __contains__(self, word):
        return word in self.key_to_index

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]


def _load_word2vec_text(path):
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        n, d = map(int, f.readline().split())
        words, vecs = [], np.empty((n, d), dtype=np.float32)
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs[i] = np.asarray(parts[1:], dtype=np.float32)
    return _PlainKeyedVectors(words, vecs)


_MODEL_CACHE = {}


def load_model():
    """Load glove-wiki-gigaword-50.

    1. gensim.downloader.load(MODEL_NAME)  (standard route)
    2. fallback: the downloaded file glove-wiki-gigaword-50.gz on disk
       (read with gensim if available, else with a small plain loader).
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]
    model = None
    try:
        import gensim.downloader as api
        model = api.load(MODEL_NAME)
    except Exception:
        candidates = [
            os.environ.get("GLOVE_PATH", ""),
            os.path.join(HERE, f"{MODEL_NAME}.gz"),
            os.path.join(HERE, "..", "data", f"{MODEL_NAME}.gz"),
            os.path.expanduser(f"~/gensim-data/{MODEL_NAME}/{MODEL_NAME}.gz"),
        ]
        path = next((p for p in candidates if p and os.path.exists(p)), None)
        if path is None:
            raise RuntimeError(
                "Could not load glove-wiki-gigaword-50. Install gensim "
                "(pip install gensim) or set GLOVE_PATH to the .gz file."
            )
        try:
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(path, binary=False)
        except ImportError:
            model = _load_word2vec_text(path)
    _MODEL_CACHE["model"] = model
    return model


# ---------------------------------------------------------------------------
# Part C -- preprocessing
# ---------------------------------------------------------------------------
def preprocess(text, stopwords=STOPWORDS):
    """Steps 1-5 of Part C, in the required order.

    Returns (raw_token_count, tokens) where tokens = x_1, ..., x_m.
    Step 6 (vocabulary check) is done in build_text_matrix().
    """
    raw_token_count = len(text.split())          # tokens before preprocessing
    text = text.lower()                          # 1. lowercase
    text = _PUNCT_RE.sub(" ", text)              # 2. remove punctuation
    tokens = text.split()                        # 3. tokenise on whitespace
    tokens = [t for t in tokens if t not in stopwords]   # 4. stopwords
    tokens = [t for t in tokens if len(t) > 2]           # 5. length filter
    return raw_token_count, tokens


# ---------------------------------------------------------------------------
# Part D -- tag matrix and text matrix
# ---------------------------------------------------------------------------
def build_tag_matrix(model, tags):
    """
    Construct the tag matrix.

    Returns:
        tag_names: list[str] containing exactly 20 tag names.
        T:         NumPy array with shape (20, 50).

    Single-word tag  -> row = model[tag].
    Multi-word tag   -> row = arithmetic mean of the component word vectors.
    Every word used in a tag must occur in the model vocabulary; otherwise a
    ValueError is raised (the tag is never silently dropped).
    """
    tag_names = list(tags)
    if len(tag_names) != 20:
        raise ValueError(f"Exactly 20 tags are required, got {len(tag_names)}.")
    if len(set(tag_names)) != 20:
        raise ValueError("Tags must be distinct.")

    rows = []
    for tag in tag_names:
        words = tag.lower().split()
        if not words:
            raise ValueError("Empty tag.")
        for w in words:
            if w not in model.key_to_index:
                raise ValueError(f"Tag not in vocabulary: {tag!r} (word {w!r})")
        vecs = np.array([model[w] for w in words], dtype=np.float64)
        rows.append(vecs.mean(axis=0))           # k = 1 gives model[tag] itself
    T = np.vstack(rows)
    assert T.shape == (20, EMBED_DIM)
    return tag_names, T


def build_text_matrix(model, tokens):
    """
    Construct the text matrix.

    Parameters:
        tokens: preprocessed tokens (after lowercasing, punctuation removal,
                stopword removal and the length filter).

    Returns:
        in_vocab_tokens: tokens in the model vocabulary, original order and
                         repetitions preserved.
        oov_tokens:      tokens not in the model vocabulary (in order).
        W:               NumPy array (n, 50); row i = embedding of
                         in_vocab_tokens[i].
    """
    in_vocab_tokens = [t for t in tokens if t in model.key_to_index]
    oov_tokens = [t for t in tokens if t not in model.key_to_index]
    if in_vocab_tokens:
        W = np.array([model[t] for t in in_vocab_tokens], dtype=np.float64)
    else:
        W = np.empty((0, EMBED_DIM), dtype=np.float64)
    return in_vocab_tokens, oov_tokens, W


# ---------------------------------------------------------------------------
# Part E -- similarity matrix with the Vector class (reference computation)
# ---------------------------------------------------------------------------
def similarity_matrix_vector_class(W, T):
    """
    Compute the complete cosine-similarity matrix using the Vector class.

    Every entry is computed by constructing two Vector instances and calling
    cosine_similarity -- no NumPy dot/norm/matmul is used for the values.

    Returns S as a nested Python list of shape (n_words, 20), where
        S[i][j] = Vector(W[i]).cosine_similarity(Vector(T[j])).
    """
    n, m = len(W), len(T)
    S = []
    for i in range(n):
        row = []
        for j in range(m):
            w_vec = Vector(W[i].tolist())     # plain Python floats
            t_vec = Vector(T[j].tolist())
            row.append(w_vec.cosine_similarity(t_vec))
        S.append(row)
    return S


# ---------------------------------------------------------------------------
# Part F -- vectorised similarity matrix with NumPy
# ---------------------------------------------------------------------------
def similarity_matrix_numpy(W, T):
    """
    Compute all text-word/tag cosine similarities using NumPy row
    normalisation and one matrix multiplication.

    Returns S with shape (n_words, 20).
    """
    W = np.asarray(W, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    W_norm = np.linalg.norm(W, axis=1, keepdims=True)    # (n, 1)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)    # (20, 1)
    if np.any(W_norm == 0):
        raise ValueError("W contains a zero row; cosine similarity undefined.")
    if np.any(T_norm == 0):
        raise ValueError("T contains a zero row; cosine similarity undefined.")
    W_hat = W / W_norm                                   # (n, 50), unit rows
    T_hat = T / T_norm                                   # (20, 50), unit rows
    S = W_hat @ T_hat.T                                  # (n,50)(50,20) = (n,20)
    return S


# ---------------------------------------------------------------------------
# Part G -- verification
# ---------------------------------------------------------------------------
RTOL = 1e-9
ATOL = 1e-12


def verify(S_vector, S_numpy, rtol=RTOL, atol=ATOL):
    """Return (allclose_flag, max_abs_difference)."""
    S_vector = np.array(S_vector, dtype=np.float64)
    ok = bool(np.allclose(S_vector, S_numpy, rtol=rtol, atol=atol))
    max_diff = float(np.max(np.abs(S_vector - S_numpy)))
    return ok, max_diff


# ---------------------------------------------------------------------------
# Part H -- max-pool tag ranking
# ---------------------------------------------------------------------------
def rank_tags(tag_names, S, in_vocab_tokens):
    """
    For each tag j:  score[j] = max_i S[i, j].
    Return the tags sorted in descending order of score, each as a dict with
    the tag, its score, the text token that produced the maximum, and whether
    that token is identical to the tag (exact match -> similarity 1).
    """
    S = np.asarray(S)
    best_rows = np.argmax(S, axis=0)                     # arg max over words
    scores = S[best_rows, np.arange(S.shape[1])]         # max over words
    results = []
    for j, tag in enumerate(tag_names):
        word = in_vocab_tokens[best_rows[j]]
        results.append({
            "tag": tag,
            "score": float(scores[j]),
            "word": word,
            "identical": word == tag,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


def format_ranking(ranking, top=8):
    lines = [
        f"{'Rank':<6}{'Tag':<19}{'Score':<9}{'Best-matching text word':<26}Match type",
        "-" * 78,
    ]
    for r in ranking[:top]:
        kind = "identical word" if r["identical"] else "different word"
        lines.append(
            f"{r['rank']:<6}{r['tag']:<19}{r['score']:<9.4f}{r['word']:<26}{kind}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(text_path=DEFAULT_TEXT):
    print("=" * 78)
    print("Semantic Tagging with GloVe (glove-wiki-gigaword-50) and NumPy")
    print("=" * 78)

    model = load_model()
    print(f"Model loaded: {len(model.key_to_index)} words, "
          f"dimension {model[next(iter(model.key_to_index))].shape[0]}")

    # Part A / D: tags
    tag_names, T = build_tag_matrix(model, TAGS)
    multi = [t for t in tag_names if len(t.split()) > 1]
    print(f"\n[Part A] {len(tag_names)} tags: {', '.join(tag_names)}")
    print(f"         multi-word tags (mean of component vectors): {multi}")

    # Part B / C: text and preprocessing
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    raw_count, tokens = preprocess(text)
    in_vocab, oov, W = build_text_matrix(model, tokens)
    distinct_oov = sorted(set(oov))

    print(f"\n[Part B] Input file: {os.path.basename(text_path)}")
    print("\n[Part C] Token statistics")
    print(f"  Tokens before preprocessing        : {raw_count}")
    print(f"  Tokens after preprocessing (m)     : {len(tokens)}")
    print(f"  In-vocabulary tokens (n)           : {len(in_vocab)}")
    print(f"  Out-of-vocabulary tokens           : {len(oov)}")
    print(f"  Distinct out-of-vocabulary tokens  : {len(distinct_oov)}")
    print(f"    {distinct_oov}")
    print(f"  Stopwords added to the starter set : {len(ADDED_STOPWORDS)}")

    # Part D checkpoint
    print("\n[Part D] Checkpoint")
    print(f"  T.shape = {T.shape}")
    print(f"  W.shape = {W.shape}")
    if W.shape[0] == 0:
        sys.exit("ERROR: the input text contains no in-vocabulary words after "
                 "preprocessing (n = 0); the similarity matrix cannot be computed.")
    assert T.shape == (20, 50) and W.shape == (len(in_vocab), 50)

    # Part E / F
    S_vector = np.array(similarity_matrix_vector_class(W, T))
    S_numpy = similarity_matrix_numpy(W, T)
    print("\n[Part E] S_vector.shape =", S_vector.shape, " (Vector class, Python loops)")
    print("[Part F] S_numpy.shape  =", S_numpy.shape, " (W_hat @ T_hat.T)")

    # Part G
    ok, max_diff = verify(S_vector, S_numpy)
    print("\n[Part G] Verification")
    print(f"  np.allclose(S_vector, S_numpy, rtol={RTOL}, atol={ATOL}) -> {ok}")
    print(f"  np.max(np.abs(S_vector - S_numpy)) = {max_diff:.3e}")

    # Part H
    ranking = rank_tags(tag_names, S_numpy, in_vocab)
    print("\n[Part H] Top 8 tags by max-pool score  r_j = max_i S_ij\n")
    print(format_ranking(ranking, top=8))
    ident = [r["tag"] for r in ranking[:8] if r["identical"]]
    diff = [f"{r['tag']} <- {r['word']}" for r in ranking[:8] if not r["identical"]]
    print(f"\n  Top-8 tags matched by an IDENTICAL word : {ident}")
    print(f"  Top-8 tags matched by a DIFFERENT word  : {diff}")

    print("\n  Full ranking of all 20 tags (for reference):\n")
    print(format_ranking(ranking, top=20))
    return ranking


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT)
