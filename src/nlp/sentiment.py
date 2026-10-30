import re
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

_analyzer = None

def analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def clean_html(value):
    text = BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def score(text):
    s = analyzer().polarity_scores(text or "")
    c = float(s["compound"])
    label = "POSITIVO" if c >= 0.05 else ("NEGATIVO" if c <= -0.05 else "NEUTRAL")
    return {
        "sentiment_compound": c,
        "sentiment_positive": float(s["pos"]),
        "sentiment_neutral": float(s["neu"]),
        "sentiment_negative": float(s["neg"]),
        "sentiment_label": label,
    }

def mood(avg):
    if avg is None:
        return "SIN DATOS"
    if avg <= -0.20:
        return "PÁNICO"
    if avg >= 0.20:
        return "CODICIA"
    return "NEUTRAL"
