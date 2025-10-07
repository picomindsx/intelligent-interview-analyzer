import subprocess
import os
import re
import json
import math
from collections import Counter

def convert_to_wav(input_file):
    output_file = os.path.splitext(input_file)[0] + ".wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", output_file
    ], check=True)
    return output_file


# optional imports
try:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# Try sentence-transformers embeddings if available
def get_sentence_embeddings(segments):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good
        return model.encode(segments, show_progress_bar=False)
    except Exception:
        return None

def tfidf_embeddings(segments):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
    X = vec.fit_transform(segments)
    return X.toarray()

# 1) split into segments (robust against missing punctuation)
def split_to_segments(text, min_words=3, max_words=35):
    text = text.strip()
    # If the transcript already has newlines per utterance, prefer that
    if '\n' in text and sum(1 for _ in text.splitlines()) > 3:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines

    # Try sentence split using punctuation
    segments = re.split(r'(?<=[.?!])\s+', text)
    segments = [s.strip() for s in segments if s.strip()]

    # If segments are too long or we have only 1 segment -> split by windows of words
    too_long = any(len(s.split()) > max_words for s in segments) or len(segments) < 2
    if too_long:
        words = text.split()
        segments = []
        i = 0
        while i < len(words):
            j = min(i + 20, len(words))  # window size ~20 words (adjustable)
            seg = " ".join(words[i:j]).strip()
            segments.append(seg)
            i = j
    # Merge tiny segments
    merged = []
    for s in segments:
        if merged and len(s.split()) < min_words:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged

# 2) get embeddings with fallback
def embed_segments(segments):
    emb = get_sentence_embeddings(segments)
    if emb is not None:
        return emb
    # fallback to TF-IDF (requires sklearn)
    if HAVE_SK:
        return tfidf_embeddings(segments)
    raise RuntimeError("No embedding method available. Install sentence-transformers or scikit-learn.")

# 3) estimate number of speakers (optional)
def estimate_num_speakers(embeddings, max_speakers=4):
    if not HAVE_SK:
        return 2  # fallback guess
    best_k = 2
    best_score = -1
    n_candidates = min(max_speakers, max(2, int(len(embeddings) / 4) + 1))
    for k in range(2, n_candidates + 1):
        try:
            cl = KMeans(n_clusters=k, n_init=8, random_state=0).fit(embeddings)
            labels = cl.labels_
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k

# 4) cluster
def cluster_embeddings(embeddings, n_speakers):
    if len(embeddings) <= n_speakers:
        # trivial: assign each to its own speaker if few segments
        return list(range(len(embeddings)))
    if HAVE_SK:
        # Agglomerative often gives more stable small-cluster results
        cl = AgglomerativeClustering(n_clusters=n_speakers)
        labels = cl.fit_predict(embeddings)
        return labels.tolist()
    # fallback trivial assignment
    return [i % n_speakers for i in range(len(embeddings))]

# 5) smoothing: median filter + merge very short segments
def smooth_labels(labels, window=3):
    if window <= 1:
        return labels
    smoothed = labels.copy()
    n = len(labels)
    for i in range(n):
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)
        window_vals = labels[start:end]
        smoothed[i] = Counter(window_vals).most_common(1)[0][0]
    return smoothed

def merge_short_segments(segments, labels, min_words=4):
    segs, labs = segments[:], labels[:]
    i = 0
    while i < len(segs):
        if len(segs[i].split()) < min_words and len(segs) > 1:
            # prefer merging with neighbor that has same label or is longer
            if i > 0:
                segs[i-1] = segs[i-1] + " " + segs[i]
                del segs[i]; del labs[i]
                i = max(0, i-1)
            else:
                segs[i] = segs[i] + " " + segs[i+1]
                del segs[i+1]; del labs[i+1]
        else:
            i += 1
    return segs, labs

# Full pipeline
def speaker_split_text(text, n_speakers=None, max_speakers=4):
    segments = split_to_segments(text)
    embeddings = embed_segments(segments)
    if n_speakers is None:
        n_speakers = estimate_num_speakers(embeddings, max_speakers=max_speakers)
    labels = cluster_embeddings(embeddings, n_speakers)
    labels = smooth_labels(labels, window=3)
    segments, labels = merge_short_segments(segments, labels, min_words=3)
    labels = smooth_labels(labels, window=3)

    # normalize mapping cluster id -> Speaker A/B/C...
    unique = sorted(list(dict.fromkeys(labels)))
    mapping = {cid: f"Speaker {chr(65 + i)}" for i, cid in enumerate(unique)}
    output = []
    for seg, lab in zip(segments, labels):
        output.append({"speaker": mapping[lab], "text": seg.strip()})
    return output
