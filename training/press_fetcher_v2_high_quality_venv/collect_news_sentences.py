#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_news_sentences.py

RSS/Atom → phrases JSONL (1 phrase/ligne).

Sortie JSONL :
  {"source":"...","lang":"fr","text":"..."}

Dépendances: feedparser, aiohttp, blingfire (optionnel mais recommandé), langid (optionnel)

Exemple:
  python collect_news_sentences.py --config feeds.json --out out.jsonl --target 200000 --langs fr,en,de,it,es,pt
"""

import argparse
import asyncio
import hashlib
import json
import re
from typing import List, Optional

import aiohttp
import feedparser

try:
    from blingfire import text_to_sentences
except Exception:
    text_to_sentences = None

try:
    import langid
except Exception:
    langid = None


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def strip_html(s: str) -> str:
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style[^>]*>.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return normalize_ws(s)


def to_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if text_to_sentences:
        out = text_to_sentences(text)
        sents = [normalize_ws(x) for x in out.split("\n")]
        return [x for x in sents if x]
    # fallback naive
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [normalize_ws(x) for x in sents if normalize_ws(x)]


def detect_lang(text: str) -> Optional[str]:
    if not langid:
        return None
    try:
        return langid.classify(text)[0]
    except Exception:
        return None


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


async def fetch_text(session: aiohttp.ClientSession, url: str, timeout_s: int) -> str:
    async with session.get(url, timeout=timeout_s, headers={"User-Agent": "rss-fetcher/2.0"}) as r:
        r.raise_for_status()
        return await r.text(errors="ignore")


def extract_entry_text(entry: dict) -> str:
    parts = []
    if entry.get("title"):
        parts.append(entry.get("title"))
    if entry.get("summary"):
        parts.append(entry.get("summary"))
    if entry.get("content"):
        try:
            for c in entry.get("content"):
                if isinstance(c, dict) and c.get("value"):
                    parts.append(c.get("value"))
        except Exception:
            pass
    return strip_html("\n".join(parts))


async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession, state: dict):
    while True:
        feed = await queue.get()
        if feed is None:
            queue.task_done()
            return
        if state["stop"]:
            queue.task_done()
            continue

        # per-feed counters
        entries_count = 0
        sentences_emitted = 0
        sentences_filtered_len = 0
        sentences_filtered_lang = 0
        sentences_deduped = 0
        feed_error = False

        try:
            raw = await fetch_text(session, feed["url"], state["timeout"]) 
            parsed = feedparser.parse(raw)
            entries = parsed.entries[: state["max_articles_per_feed"]] if state["max_articles_per_feed"] else parsed.entries
            entries_count = len(entries)
            state["entries_total"] += entries_count

            for entry in entries:
                txt = extract_entry_text(entry)
                if not txt:
                    continue
                for sent in to_sentences(txt):
                    if state["min_chars"] and len(sent) < state["min_chars"]:
                        sentences_filtered_len += 1
                        continue
                    if state["max_chars"] and len(sent) > state["max_chars"]:
                        sentences_filtered_len += 1
                        continue
                    if state["langs"]:
                        if state["strict_lang"]:
                            dl = detect_lang(sent)
                            if dl and dl not in state["langs"]:
                                sentences_filtered_lang += 1
                                continue
                        else:
                            if feed.get("lang") not in state["langs"]:
                                sentences_filtered_lang += 1
                                continue
                    h = stable_hash(feed.get("source", "?") + "|" + feed.get("lang", "?") + "|" + sent)
                    if h in state["seen"]:
                        sentences_deduped += 1
                        continue
                    state["seen"].add(h)
                    rec = {"source": feed.get("source", "unknown"), "lang": feed.get("lang", ""), "text": sent}
                    state["out_f"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    state["count"] += 1
                    sentences_emitted += 1
                    if state["target"] and state["count"] >= state["target"]:
                        state["stop"] = True
                        break
                if state["stop"]:
                    break

        except Exception as e:
            feed_error = True
            state["feed_errors"] += 1
            if state.get("verbose"):
                print(f"[ERROR] feed {feed.get('url')} -> {e}")
        finally:
            # update global counters
            state["feeds_processed"] += 1
            state["sentences_total"] += sentences_emitted
            state["sentences_filtered_len"] += sentences_filtered_len
            state["sentences_filtered_lang"] += sentences_filtered_lang
            state["sentences_deduped"] += sentences_deduped

            if state.get("verbose"):
                print(
                    f"[FEED] url={feed.get('url')} entries={entries_count} emitted={sentences_emitted} "
                    f"filtered_len={sentences_filtered_len} filtered_lang={sentences_filtered_lang} deduped={sentences_deduped} error={feed_error}"
                )
            queue.task_done()


async def main_async(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feeds = cfg.get("feeds", [])

    state = {
        "target": args.target,
        "count": 0,
        "stop": False,
        "seen": set(),
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "timeout": args.timeout,
        "max_articles_per_feed": args.max_articles_per_feed,
        "langs": set(args.langs.split(",")) if args.langs else set(),
        "strict_lang": args.strict_lang,
        "out_f": open(args.out, "w", encoding="utf-8"),
        # diagnostics
        "verbose": args.verbose,
        "feeds_processed": 0,
        "entries_total": 0,
        "sentences_total": 0,
        "sentences_filtered_len": 0,
        "sentences_filtered_lang": 0,
        "sentences_deduped": 0,
        "feed_errors": 0,
    }

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        q = asyncio.Queue()
        for feed in feeds:
            await q.put(feed)
        workers = [asyncio.create_task(worker(q, session, state)) for _ in range(args.concurrency)]
        await q.join()
        for _ in workers:
            await q.put(None)
        await asyncio.gather(*workers, return_exceptions=True)

    state["out_f"].close()

    if state.get("verbose"):
        print("\n[SUMMARY]")
        print(f"feeds_processed: {state['feeds_processed']}")
        print(f"entries_total: {state['entries_total']}")
        print(f"sentences_emitted: {state['sentences_total']}")
        print(f"sentences_filtered_len: {state['sentences_filtered_len']}")
        print(f"sentences_filtered_lang: {state['sentences_filtered_lang']}")
        print(f"sentences_deduped: {state['sentences_deduped']}")
        print(f"feed_errors: {state['feed_errors']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--langs", default="", help="ex: fr,en,de")
    ap.add_argument("--strict-lang", action="store_true", help="langid au niveau phrase")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--max-articles-per-feed", type=int, default=800)
    ap.add_argument("--verbose", action="store_true", help="mode verbeux: affiche diagnostics par flux et résumé final")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
