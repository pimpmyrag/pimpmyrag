#!/usr/bin/env python3
"""
annotate_nominal_parents_haiku_batch.py — v1.0 (v8.22)
=======================================================
Étape 3 du pipeline nominal_parent_pointer :
Envoie les phrases marquées needs_haiku=True à Claude Haiku via Batch API.

Input  : *_stanza.jsonl (sortie de stanza_inject_nominal_parents.py)
Output : *_v8.22_nominal.jsonl (dataset final avec tous les champs annotés)

Usage :
    cd training/multi-head
    source venv/bin/activate && source .secrets.env

    # Créer le batch et attendre
    python3 scripts/annotate_nominal_parents_haiku_batch.py \\
        --input  data/train_v8.21_verbfam_stanza.jsonl \\
        --output data/train_v8.22_nominal.jsonl \\
        --api-key $ANTHROPIC_API_KEY \\
        --poll-interval 60

    # Reprendre un batch existant
    python3 scripts/annotate_nominal_parents_haiku_batch.py \\
        --input  data/train_v8.21_verbfam_stanza.jsonl \\
        --output data/train_v8.22_nominal.jsonl \\
        --api-key $ANTHROPIC_API_KEY \\
        --batch-id msgbatch_...
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

# ── Prompt système Haiku (statique — envoyé une seule fois) ──────────────────
SYSTEM_PROMPT = """Tu es un annotateur linguistique expert en français. Tu analyses des relations nominales et des coréférences intraphrastiques.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ou après.
2. Utilise UNIQUEMENT les IDs de spans fournis dans "spans" (s0, s1, s2...).
3. Ne crée AUCUN nouveau span (sauf dans "suggested_missing_spans" optionnel).
4. Pour chaque arête nominale, choisis la relation UNIQUEMENT dans :
   APPOS | NMOD | POSS | AMOD | COMPOUND | SOURCE | MEDIUM | LOC | TIME | MISC
5. Pour la coréférence, score ∈ [0.0, 1.0]. Ne liste que les candidats avec score ≥ 0.10.
6. Si un candidate_edge dans l'input est correct, confirme-le avec confidence ≥ celle du candidat.
7. Si un candidate_edge est incorrect, place-le dans "rejected_edges" avec reason_code.
8. reason_code : mot-clé court en snake_case (ex: role_name_apposition, gender_number_subject_salience, wrong_parent, ambiguous_head, comm_event_source).
9. confidence ∈ [0.50, 1.00] pour nominal_edges. En dessous de 0.50, place dans rejected_edges.
10. event_links uniquement si relation évidente (ATTRIBUTION avec marqueur explicite "comme le souligne", "selon", etc.).
11. Ne jamais inventer un parent qui n'est pas dans la liste des spans fournis.

FORMAT DE RÉPONSE JSON :
{
  "nominal_edges": [
    {"child": "sX", "parent": "sY", "relation": "APPOS", "confidence": 0.95, "reason_code": "role_name_apposition"}
  ],
  "coref_edges": [
    {"mention": "sX", "target": "sY", "confidence": 0.86, "reason_code": "gender_number_subject_salience"}
  ],
  "event_links": [
    {"source_event": "sX", "target_event": "sY", "relation": "ATTRIBUTION", "confidence": 0.82, "reason_code": "comme_le_souligne"}
  ],
  "rejected_edges": [
    {"child": "sX", "parent": "sY", "reason_code": "wrong_parent"}
  ],
  "suggested_missing_spans": []
}"""


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_haiku_request(row: dict) -> dict:
    spans = row.get("spans", [])
    candidate_edges = row.get("candidate_edges", [])

    spans_payload = []
    span_id_map = {}

    for i, sp in enumerate(spans):
        sid = f"s{i}"
        span_id_map[(sp["start"], sp["end"])] = sid

        spans_payload.append({
            "id": sid,
            # 🔥 optimisation : trimming texte inutile
            "text": sp.get("text", "")[:80],  # limit tokens
            "start": sp["start"],
            "end": sp["end"],
            "label": sp["label"],
            "svo_role": sp.get("svo_role", "NONE"),
        })

    edges_payload = []
    for e in candidate_edges:
        child_id  = span_id_map.get((e["child_start"], e["child_end"]))
        parent_id = span_id_map.get(
            (e.get("parent_start"), e.get("parent_end"))
        ) if e.get("parent_start") else None

        if child_id:
            edge = {
                "child": child_id,
                "relation": e["relation"],
                "confidence": e["confidence"],
            }
            if parent_id:
                edge["parent"] = parent_id
            edges_payload.append(edge)

    # 🔥 IMPORTANT : prompt compact (moins de tokens)
    user_content = json.dumps({
        "text": row["text"][:500],  # clip sensible
        "spans": spans_payload,
        "candidate_edges": edges_payload,
    }, ensure_ascii=False)

    return {
        "custom_id": str(row["id"]),
        "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 500,
            "temperature": 0.0,
            # Prompt caching: bloc statique mutualisé entre requêtes du batch.
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
                            "text": user_content
                        }
                    ]
                }
            ],
        }
    }


def submit_batch(requests: list[dict], api_key: str,
                 requests_file: str) -> str:
    """Envoie le batch à l'API Anthropic et retourne le batch_id."""
    with open(requests_file, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"📤 Envoi batch de {len(requests):,} requêtes (prompt caching activé)...", flush=True)

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
            "content-type": "application/json",
        },
        json={"requests": requests},
        timeout=120,
    )
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"✅ Batch soumis : {batch_id}", flush=True)
    return batch_id


def poll_batch(batch_id: str, api_key: str, poll_interval: int = 60) -> str:
    """Poll jusqu'à ce que le batch soit terminé. Retourne l'URL des résultats."""
    print(f"⏳ Polling batch {batch_id}...", flush=True)
    while True:
        resp = httpx.get(
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta":    "message-batches-2024-09-24",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data["processing_status"]
        counts = data.get("request_counts", {})
        print(f"  {status} — {counts}", flush=True)
        if status == "ended":
            return data["results_url"]
        time.sleep(poll_interval)


def download_results(results_url: str, api_key: str) -> list[dict]:
    """Télécharge et parse les résultats du batch."""
    resp = httpx.get(
        results_url,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta":    "message-batches-2024-09-24",
        },
        timeout=120,
        follow_redirects=True,
    )
    resp.raise_for_status()
    results = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def parse_haiku_result(result: dict) -> dict | None:
    """
    Parse la réponse Haiku pour un custom_id.
    Retourne le JSON parsé ou None si erreur.
    """
    if result.get("result", {}).get("type") != "succeeded":
        return None
    try:
        content = result["result"]["message"]["content"]
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        else:
            text = str(content)
        # Extraire le JSON (peut être entouré de ```json...```)
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return None


def apply_haiku_result(row: dict, haiku_out: dict) -> dict:
    """
    Applique les résultats Haiku sur les spans de la phrase.
    """
    spans = row.get("spans", [])
    existing_edges = row.get("candidate_edges", [])

    # Reconstruire le mapping sX → (start, end)
    id_to_span: dict[str, dict] = {}
    for i, sp in enumerate(spans):
        id_to_span[f"s{i}"] = sp

    # Fusionner les arêtes nominales Haiku
    nominal_edges = haiku_out.get("nominal_edges", [])
    # Remplacer les arêtes conflictuelles par les résultats Haiku
    haiku_by_child: dict[tuple, dict] = {}
    for edge in nominal_edges:
        child_sp  = id_to_span.get(edge.get("child", ""))
        parent_sp = id_to_span.get(edge.get("parent", ""))
        if child_sp is None or parent_sp is None:
            continue
        haiku_by_child[(child_sp["start"], child_sp["end"])] = {
            "child_start":  child_sp["start"],
            "child_end":    child_sp["end"],
            "child_text":   child_sp.get("text", ""),
            "parent_start": parent_sp["start"],
            "parent_end":   parent_sp["end"],
            "parent_text":  parent_sp.get("text", ""),
            "relation":     edge["relation"],
            "confidence":   edge.get("confidence", 0.8),
            "source":       "haiku",
        }

    # Coref candidates
    coref_edges = haiku_out.get("coref_edges", [])
    coref_by_mention: dict[tuple, list] = {}
    for ce in coref_edges:
        mention_sp = id_to_span.get(ce.get("mention", ""))
        target_sp  = id_to_span.get(ce.get("target", ""))
        if mention_sp is None or target_sp is None:
            continue
        key = (mention_sp["start"], mention_sp["end"])
        coref_by_mention.setdefault(key, []).append({
            "target_start": target_sp["start"],
            "target_end":   target_sp["end"],
            "target_text":  target_sp.get("text", ""),
            "score":        ce.get("confidence", 0.5),
            "source":       "haiku",
        })

    # Réinjecter dans les spans
    annotated_spans = []
    for sp in spans:
        sp_copy = dict(sp)
        key = (sp["start"], sp["end"])

        # Nominal parent depuis Haiku (priorité absolue sur rule/stanza)
        if key in haiku_by_child:
            e = haiku_by_child[key]
            sp_copy["nominal_parent_start"]      = e["parent_start"]
            sp_copy["nominal_parent_end"]        = e["parent_end"]
            sp_copy["nominal_parent_text"]       = e["parent_text"]
            sp_copy["nominal_relation"]          = e["relation"]
            sp_copy["nominal_parent_confidence"] = e["confidence"]
            sp_copy["nominal_parent_source"]     = "haiku"

        # Coref candidates
        if key in coref_by_mention:
            sp_copy["coref_candidates"] = sorted(
                coref_by_mention[key], key=lambda c: c["score"], reverse=True
            )[:5]  # top-5

        annotated_spans.append(sp_copy)

    # Event links au niveau de la phrase
    event_links = haiku_out.get("event_links", [])
    event_links_out = []
    for el in event_links:
        src_sp = id_to_span.get(el.get("source_event", ""))
        tgt_sp = id_to_span.get(el.get("target_event", ""))
        if src_sp and tgt_sp:
            event_links_out.append({
                "source_event_start": src_sp["start"],
                "target_event_start": tgt_sp["start"],
                "relation":           el.get("relation", "ATTRIBUTION"),
                "confidence":         el.get("confidence", 0.7),
                "source":             "haiku",
            })

    # Mise à jour finale de l'arête candidate avec les résultats Haiku
    final_edges = []
    seen = set()
    for e in haiku_by_child.values():
        key = (e["child_start"], e["child_end"])
        if key not in seen:
            seen.add(key)
            final_edges.append(e)
    # Garder les arêtes rule/stanza non overridées
    for e in existing_edges:
        key = (e["child_start"], e["child_end"])
        if key not in seen:
            seen.add(key)
            final_edges.append(e)

    result = dict(row)
    result["spans"] = annotated_spans
    result["candidate_edges"] = final_edges
    if event_links_out:
        result["event_links"] = event_links_out
    result.pop("needs_haiku", None)  # nettoyage
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="*_stanza.jsonl")
    parser.add_argument("--output",  required=True, help="*_v8.22_nominal.jsonl")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--secrets-env", default=DEFAULT_SECRETS_ENV,
                        help="Chemin vers .secrets.env (chargé auto si présent)")
    parser.add_argument("--batch-id", default=None,
                        help="ID d'un batch existant à reprendre")
    parser.add_argument("--requests-file", default=None,
                        help="Fichier JSONL des requêtes (défaut: <output>.requests.jsonl)")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--force-haiku-all", action="store_true",
                        help="Ignore needs_haiku et envoie toutes les phrases à Haiku")
    args = parser.parse_args()

    secrets_path = Path(args.secrets_env)
    if secrets_path.exists():
        load_dotenv(secrets_path, override=False)

    if not args.api_key:
        args.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not args.api_key:
        print("❌ ANTHROPIC_API_KEY manquant", file=sys.stderr)
        sys.exit(1)

    requests_file = args.requests_file or (args.output.replace(".jsonl", "_requests.jsonl"))

    # Charger toutes les phrases
    all_rows = list(load_jsonl(args.input))
    if args.force_haiku_all:
        haiku_rows = list(all_rows)
        non_haiku_rows = []
    else:
        haiku_rows = [r for r in all_rows if r.get("needs_haiku")]
        non_haiku_rows = [r for r in all_rows if not r.get("needs_haiku")]

    print(f"📊 Total : {len(all_rows):,} phrases")
    print(f"   Haiku : {len(haiku_rows):,} ({len(haiku_rows)/max(1,len(all_rows))*100:.1f}%)")
    print(f"   Direct : {len(non_haiku_rows):,}", flush=True)

    # Si pas de phrases Haiku → écrire directement
    if not haiku_rows:
        print("ℹ️  Aucune phrase nécessitant Haiku → écriture directe", flush=True)
        for row in all_rows:
            row.pop("needs_haiku", None)
        write_jsonl(args.output, all_rows)
        print(f"✅ Output : {args.output}")
        return

    # Soumettre ou reprendre le batch
    if args.batch_id:
        batch_id = args.batch_id
        print(f"♻️  Reprise batch : {batch_id}", flush=True)
    else:
        requests = [build_haiku_request(r) for r in haiku_rows]
        batch_id = submit_batch(requests, args.api_key, requests_file)

        # Sauvegarder le batch_id
        batch_id_file = Path(requests_file).with_suffix(".batch_id")
        batch_id_file.write_text(batch_id)
        print(f"   Batch ID sauvegardé : {batch_id_file}", flush=True)

    # Attendre les résultats
    results_url = poll_batch(batch_id, args.api_key, args.poll_interval)
    print(f"📥 Téléchargement résultats...", flush=True)
    results = download_results(results_url, args.api_key)
    print(f"   {len(results):,} résultats reçus", flush=True)

    # Indexer les résultats par custom_id
    results_by_id = {r["custom_id"]: r for r in results}

    # Fusionner Haiku + non-Haiku dans l'ordre original
    haiku_ids = {str(r["id"]) for r in haiku_rows}
    output_rows = []
    n_haiku_applied = 0
    n_haiku_failed  = 0

    for row in all_rows:
        row.pop("needs_haiku", None)
        row_id = str(row["id"])
        if row_id not in haiku_ids:
            output_rows.append(row)
            continue

        # Appliquer les résultats Haiku
        result = results_by_id.get(row_id)
        if result is None:
            output_rows.append(row)
            n_haiku_failed += 1
            continue

        haiku_out = parse_haiku_result(result)
        if haiku_out is None:
            output_rows.append(row)
            n_haiku_failed += 1
            continue

        output_rows.append(apply_haiku_result(row, haiku_out))
        n_haiku_applied += 1

    write_jsonl(args.output, output_rows)

    print(f"\n✅ Terminé : {len(output_rows):,} phrases écrites")
    print(f"   Haiku appliqué : {n_haiku_applied:,}")
    print(f"   Haiku échoué (kept rule/stanza) : {n_haiku_failed:,}")
    print(f"   Output : {args.output}")


if __name__ == "__main__":
    main()

