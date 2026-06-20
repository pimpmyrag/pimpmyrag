#!/usr/bin/env python3
"""
repair_spans_haiku_batch.py
===========================

Pipeline Haiku-first pour nettoyer les spans d'un dataset JSONL :
- ajout de verb_trigger manquants
- suppression de doublons/faux positifs
- correction des frontières (notamment groupes nominaux)
- correction des rôles SVO (évite les OBLIQUE "à plat")

Le script fonctionne par passes. Chaque passe lit les spans actuels, envoie une requête
par phrase à Claude Haiku (Batch API), puis remplace la liste de spans par la sortie validée.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


DEFAULT_SECRETS_ENV = "/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/.secrets.env"

VALID_LABELS = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_inst_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_weapon", "hint_vehicle", "hint_substance", "hint_food", "hint_infra", "hint_tool",
    "hint_object_generic", "hint_object_name", "hint_event_nominal", "hint_event_named",
    "hint_time_date", "hint_time_clock", "hint_time_duration", "hint_measure", "hint_percentage",
    "hint_count", "hint_money", "hint_rate", "hint_work_of_art", "hint_law", "hint_document",
    "hint_disease", "hint_language", "hint_inst_role", "hint_doctrine", "hint_state",
    "hint_notion", "hint_work_generic", "hint_field", "verb_trigger", "pron_subj", "pron_obj",
    "hint_pron_subj",
}

VALID_ROLES = {
    "SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE",
    "OBLIQUE_ADVERSARY", "OBLIQUE_BENEFICIARY", "OBLIQUE_COMITATIVE",
    "OBLIQUE_DOMAIN", "OBLIQUE_SOURCE", "APPOS", "NONE",
}

SYSTEM_PROMPT = """Tu es un expert en annotation NER+SVO du français.
Tu dois RENVOYER UNE VERSION CORRIGÉE COMPLÈTE des spans d'une phrase.

Objectifs prioritaires :
1) Ajouter les verb_trigger manquants (verbes ou participes verbaux gouvernant des arguments).
2) Corriger les frontières de spans : span minimal lexical, pas de déterminant inutile.
3) Supprimer les faux positifs et les doublons (y compris doublons à frontières différentes).
4) Corriger les rôles SVO :
   - Évite les OBLIQUE "à plat" non reliés à un vrai prédicat.
   - Mets NONE quand le lien verbal n'est pas clair.
   - Garde OBLIQUE_* seulement si l'indice syntaxico-sémantique est explicite.
5) Conserver les labels valides du schéma.

Règles strictes :
- Réponds UNIQUEMENT avec un objet JSON valide.
- Retourne la liste complète finale dans "spans".
- Chaque span doit contenir: start, end, text, label.
- Ajoute svo_role seulement quand pertinent (sinon omettre ou NONE).
- text doit correspondre EXACTEMENT à text[start:end].
- Interdit: créer des offsets hors phrase.

Format JSON de réponse:
{
  "spans": [
    {"start": 10, "end": 18, "text": "Macron", "label": "hint_person_name", "svo_role": "SUBJECT"},
    {"start": 22, "end": 28, "text": "signé", "label": "verb_trigger"}
  ]
}
"""


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_span_for_prompt(sp: dict) -> dict:
    out = {
        "start": sp.get("start"),
        "end": sp.get("end"),
        "text": sp.get("text", ""),
        "label": sp.get("label", ""),
    }
    if sp.get("svo_role"):
        out["svo_role"] = sp.get("svo_role")
    return out


def build_user_content(row: dict) -> str:
    payload = {
        "id": str(row.get("id", "")),
        "text": row.get("text", "")[:1200],
        "current_spans": [_clean_span_for_prompt(sp) for sp in row.get("spans", [])],
        "instruction": "Return corrected final spans for this sentence.",
    }
    return json.dumps(payload, ensure_ascii=False)


def build_request(row: dict, model: str, max_tokens: int) -> dict:
    return {
        "custom_id": str(row["id"]),
        "params": {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_user_content(row),
                        }
                    ],
                }
            ],
        },
    }


def create_requests_file(rows: list[dict], path: str, model: str, max_tokens: int):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            req = build_request(row, model=model, max_tokens=max_tokens)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def read_requests(path: str) -> list[dict]:
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reqs.append(json.loads(line))
    return reqs


def submit_batch(api_key: str, requests_file: str) -> str:
    requests = read_requests(requests_file)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
        "content-type": "application/json",
    }
    print(f"📤 Envoi {len(requests):,} requêtes Batch API...", flush=True)
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers=headers,
        json={"requests": requests},
        timeout=120,
    )
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"✅ Batch créé: {batch_id}", flush=True)
    return batch_id


def poll_batch(api_key: str, batch_id: str, poll_interval: int) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    print(f"⏳ Polling {batch_id}...", flush=True)
    while True:
        resp = httpx.get(
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("processing_status", "unknown")
        counts = data.get("request_counts", {})
        print(f"  {status} — {counts}", flush=True)
        if status == "ended":
            return data["results_url"]
        time.sleep(poll_interval)


def download_results(api_key: str, results_url: str) -> list[dict]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    resp = httpx.get(results_url, headers=headers, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    out = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def parse_message_json(result: dict) -> dict | None:
    if result.get("result", {}).get("type") != "succeeded":
        return None
    try:
        parts = result["result"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return None


def repair_offset(text: str, span_text: str, hint_start: int, hint_end: int, window: int = 50):
    if not span_text:
        return None
    lo = max(0, hint_start - window)
    hi = min(len(text), hint_end + window)
    idx = text.find(span_text, lo, hi)
    if idx >= 0:
        return idx, idx + len(span_text)
    idx = text.find(span_text)
    if idx >= 0:
        return idx, idx + len(span_text)
    return None


def normalize_spans(text: str, spans: list[dict]) -> list[dict]:
    cleaned = []
    for sp in spans:
        start = sp.get("start")
        end = sp.get("end")
        label = sp.get("label")
        span_text = sp.get("text", "")

        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        if label not in VALID_LABELS:
            continue

        if text[start:end] != span_text:
            fixed = repair_offset(text, span_text, start, end)
            if not fixed:
                continue
            start, end = fixed
            span_text = text[start:end]

        out = {
            "start": start,
            "end": end,
            "text": span_text,
            "label": label,
        }

        role = sp.get("svo_role")
        if isinstance(role, str) and role in VALID_ROLES and role != "NONE":
            out["svo_role"] = role
        elif role == "NONE":
            out["svo_role"] = "NONE"

        cleaned.append(out)

    # Déduplication stricte
    seen = set()
    dedup = []
    for sp in sorted(cleaned, key=lambda x: (x["start"], x["end"], x["label"])):
        key = (sp["start"], sp["end"], sp["label"], sp.get("svo_role", ""))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(sp)

    # Dédup verb_trigger par overlap fort: garde le span le plus court
    verb = [s for s in dedup if s["label"] == "verb_trigger"]
    other = [s for s in dedup if s["label"] != "verb_trigger"]
    verb_sorted = sorted(verb, key=lambda x: (x["start"], x["end"] - x["start"]))
    verb_out = []
    for sp in verb_sorted:
        keep = True
        for k in verb_out:
            inter = min(sp["end"], k["end"]) - max(sp["start"], k["start"])
            if inter <= 0:
                continue
            # Si overlap et même ancrage initial/final, on privilégie la frontière la plus courte.
            if sp["start"] == k["start"] or sp["end"] == k["end"]:
                len_sp = sp["end"] - sp["start"]
                len_k = k["end"] - k["start"]
                if len_sp < len_k:
                    k.update(sp)
                keep = False
                break
        if keep:
            verb_out.append(sp)

    return sorted(other + verb_out, key=lambda x: (x["start"], x["end"], x["label"]))


def apply_results(rows: list[dict], results: list[dict]) -> tuple[list[dict], int, int]:
    by_id = {str(r.get("custom_id")): r for r in results}
    out_rows = []
    n_ok = 0
    n_fallback = 0

    for row in rows:
        row_id = str(row["id"])
        result = by_id.get(row_id)
        if not result:
            out_rows.append(row)
            n_fallback += 1
            continue

        parsed = parse_message_json(result)
        if not parsed or not isinstance(parsed.get("spans"), list):
            out_rows.append(row)
            n_fallback += 1
            continue

        new_spans = normalize_spans(row.get("text", ""), parsed["spans"])
        merged = dict(row)
        merged["spans"] = new_spans
        merged.pop("needs_haiku", None)
        out_rows.append(merged)
        n_ok += 1

    return out_rows, n_ok, n_fallback


def run_one_pass(
    rows: list[dict],
    api_key: str,
    requests_file: str,
    poll_interval: int,
    model: str,
    max_tokens: int,
    batch_id: str | None = None,
) -> tuple[list[dict], int, int]:
    if not batch_id:
        create_requests_file(rows, requests_file, model=model, max_tokens=max_tokens)
        batch_id = submit_batch(api_key, requests_file)
        Path(requests_file).with_suffix(".batch_id").write_text(batch_id)

    results_url = poll_batch(api_key, batch_id, poll_interval)
    results = download_results(api_key, results_url)
    return apply_results(rows, results)


def main():
    parser = argparse.ArgumentParser(description="Nettoie les spans via Claude Haiku Batch API")
    parser.add_argument("--input", required=True, help="JSONL source")
    parser.add_argument("--output", required=True, help="JSONL destination")
    parser.add_argument("--api-key", default="", help="API key Anthropic")
    parser.add_argument("--secrets-env", default=DEFAULT_SECRETS_ENV,
                        help="Chemin vers .secrets.env (chargement auto)")
    parser.add_argument("--requests-file", default=None,
                        help="Fichier JSONL des requêtes (défaut dérivé de --output)")
    parser.add_argument("--batch-id", default=None, help="Reprendre un batch existant (1 passe)")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--passes", type=int, default=1, help="Nombre de passes Haiku")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Limiter le volume pour debug")
    args = parser.parse_args()

    env_path = Path(args.secrets_env)
    if env_path.exists():
        load_dotenv(env_path, override=False)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY manquant (option --api-key ou .secrets.env)", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(args.input)
    if args.max_sentences is not None:
        rows = rows[: max(0, args.max_sentences)]

    if not rows:
        print("ℹ️ Aucun enregistrement à traiter")
        write_jsonl(args.output, [])
        return

    print(f"📊 Phrases à traiter: {len(rows):,}")
    print(f"🔁 Passes: {args.passes}")

    if args.batch_id and args.passes != 1:
        print("❌ --batch-id n'est supporté que pour --passes 1", file=sys.stderr)
        sys.exit(1)

    current_rows = rows
    total_ok = 0
    total_fallback = 0

    for p in range(1, args.passes + 1):
        req_file = args.requests_file or args.output.replace(".jsonl", f"_pass{p}_requests.jsonl")
        print(f"\n=== Passe {p}/{args.passes} ===")
        pass_batch_id = args.batch_id if p == 1 else None
        current_rows, n_ok, n_fallback = run_one_pass(
            current_rows,
            api_key=api_key,
            requests_file=req_file,
            poll_interval=args.poll_interval,
            model=args.model,
            max_tokens=args.max_tokens,
            batch_id=pass_batch_id,
        )
        total_ok += n_ok
        total_fallback += n_fallback
        print(f"✅ Passe {p}: appliqué={n_ok:,} fallback={n_fallback:,}")

    write_jsonl(args.output, current_rows)
    print("\n🎉 Terminé")
    print(f"   Output: {args.output}")
    print(f"   Phrases finales: {len(current_rows):,}")
    print(f"   Total appliqué: {total_ok:,}")
    print(f"   Total fallback: {total_fallback:,}")


if __name__ == "__main__":
    main()

