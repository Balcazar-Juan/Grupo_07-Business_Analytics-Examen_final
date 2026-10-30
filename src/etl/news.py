import calendar, hashlib, re
from datetime import datetime, timezone
import feedparser
from pymongo import UpdateOne
from src.config import settings
from src.db import get_db
from src.nlp.sentiment import clean_html, score

def published(entry):
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        p = getattr(entry, attr, None)
        if p:
            return datetime.fromtimestamp(calendar.timegm(p), tz=timezone.utc)
    return datetime.now(timezone.utc)

def mentions(text, assets):
    found = []
    for a in assets:
        name = str(a.get("name") or "")
        symbol = str(a.get("symbol") or "")
        hit_name = bool(name and re.search(rf"\b{re.escape(name)}\b", text, re.I))
        hit_symbol = len(symbol) >= 3 and bool(
            re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text, re.I)
        )
        if hit_name or hit_symbol:
            found.append(a["coin_id"])
    return found

def refresh_news():
    db = get_db()
    assets = list(db.assets.find(
        {"active_top_n": True},
        {"_id": 0, "coin_id": 1, "name": 1, "symbol": 1}
    ))
    feeds = {
        "CoinDesk": settings.rss_coindesk,
        "Cointelegraph": settings.rss_cointelegraph,
        "Decrypt": settings.rss_decrypt,
    }
    now = datetime.now(timezone.utc)
    ops, counts = [], {}
    for source, url in feeds.items():
        parsed = feedparser.parse(url)
        entries = parsed.entries[:50]
        counts[source] = len(entries)
        for entry in entries:
            title = clean_html(getattr(entry, "title", ""))
            summary = clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            link = str(getattr(entry, "link", "")).strip()
            pub = published(entry)
            key = link or f"{source}|{title}|{pub.isoformat()}"
            news_id = hashlib.sha256(key.encode()).hexdigest()
            text = f"{title}. {summary}"
            doc = {
                "news_id": news_id,
                "source": source,
                "title": title,
                "summary": summary,
                "url": link,
                "published_at": pub,
                "mentioned_coin_ids": mentions(text, assets),
                "ingested_at": now,
                **score(text),
            }
            ops.append(UpdateOne({"news_id": news_id}, {"$set": doc}, upsert=True))
    if ops:
        db.news.bulk_write(ops, ordered=False)
    return {"rows": len(ops), "per_source": counts}
