"""
preprocessing.py
----------------
NLP text cleaning pipeline for the content moderation system.

Works WITHOUT any NLTK downloads — uses bundled stopwords and
simple suffix-based lemmatization as fallbacks.
If NLTK data IS available, it automatically uses the better versions.

Pipeline:
  raw text -> lowercase -> remove URLs -> remove special chars
           -> tokenize -> remove stopwords -> lemmatize -> clean string
"""

import re

# ── Bundled English stopwords (no NLTK download needed) ─────────────────────
_BUNDLED_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","couldn","did","didn","do","does","doesn",
    "doing","don","down","during","each","few","for","from","further","get","got",
    "had","hadn","has","hasn","have","haven","having","he","her","here","hers",
    "herself","him","himself","his","how","i","if","in","into","is","isn","it",
    "its","itself","just","ll","m","ma","me","mightn","more","most","mustn","my",
    "myself","needn","o","of","off","on","once","only","or","other","our","ours",
    "ourselves","out","over","own","re","s","same","shan","she","should","shouldn",
    "so","some","such","t","than","that","the","their","theirs","them","themselves",
    "then","there","these","they","this","those","through","to","too","under",
    "until","up","ve","very","was","wasn","we","were","weren","what","when",
    "where","which","while","who","whom","why","will","with","won","wouldn",
    "y","your","yours","yourself","yourselves","said","also","would","could",
    "one","two","three","like","even","still","well","back","use","know","see",
}

# Try NLTK stopwords; silently fall back to bundled list
try:
    import nltk
    from nltk.corpus import stopwords as _nltk_sw
    _stop_words = set(_nltk_sw.words("english"))
except Exception:
    _stop_words = _BUNDLED_STOPWORDS

# Words that look like stopwords but carry strong toxic signal — never remove
_TOXIC_KEEP = {"not", "no", "never", "against", "hate", "kill", "die", "you"}
_FINAL_STOP_WORDS = _stop_words - _TOXIC_KEEP


# ── Lemmatization ─────────────────────────────────────────────────────────────

def _simple_lemmatize(word: str) -> str:
    """Rule-based suffix stripping — fallback when NLTK WordNet unavailable."""
    for suffix, replacement in [
        ("inging", "ing"), ("tion", "te"), ("ness", ""),
        ("ies", "y"), ("ied", "y"), ("ing", ""),
        ("est", ""), ("er", ""), ("es", ""), ("s", ""),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)] + replacement
    return word


def _lemmatize(word: str) -> str:
    """Use NLTK WordNet lemmatizer if available, else use simple rules."""
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(word)
    except Exception:
        return _simple_lemmatize(word)


def _tokenize(text: str) -> list:
    """Use NLTK word_tokenize if available, else split on whitespace."""
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return text.split()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline for a single comment.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Keep only a-z + spaces
      4. Tokenize
      5. Remove stopwords (keep toxic-signal words: hate, kill, die, etc.)
      6. Lemmatize
      7. Rejoin into cleaned string

    Returns:
        Space-separated cleaned string ready for TF-IDF vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = _tokenize(text)

    result = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in _FINAL_STOP_WORDS:
            continue
        result.append(_lemmatize(token))

    return " ".join(result)


def preprocess_series(series):
    """Apply clean_text to a pandas Series. Nulls become empty strings."""
    return series.fillna("").apply(clean_text)
