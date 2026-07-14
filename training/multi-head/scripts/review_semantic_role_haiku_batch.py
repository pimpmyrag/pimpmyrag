#!/usr/bin/env python3
"""
review_semantic_role_haiku_batch.py — Révision des rôles sémantiques non résolus
(OBLIQUE_UNRESOLVED) via Claude Haiku Batch API.

Contexte :
  Le mapper heuristique (build_multitask_dataset.py / annotate_nominal_parents.py)
  dérive "semantic_role" depuis svo_role + verb_family + nominal_relation. Certains
  cas restent non résolus (OBLIQUE_UNRESOLVED = SEMANTIC_ROLE_SKIP_ID, non supervisé
  dans la loss) faute de règle fiable — typiquement des OBLIQUE avec verb_family
  ambigu, ou des relations nominales AMOD/COMPOUND/MISC.

  Ce script demande à Claude de trancher, pour chaque span OBLIQUE_UNRESOLVED,
  un rôle sémantique parmi la taxonomie SEMANTIC_ROLE_LABELS (ou de confirmer
  qu'aucun rôle fiable n'est déterminable → reste OBLIQUE_UNRESOLVED).

Usage:
  # 1) Générer les requêtes batch
  python3 scripts/review_semantic_role_haiku_batch.py \\
      --input  data/train_v8.24_nominal.jsonl \\
      --output data/train_v8.25.jsonl \\
      --requests-file data/_review_semantic_role_train_requests.jsonl \\
      --api-key $ANTHROPIC_API_KEY \\
      --batch-size 8 \\
      --model claude-haiku-4-5

  # 2) Reprendre un batch existant (poll + application)
  python3 scripts/review_semantic_role_haiku_batch.py \\
      --input data/train_v8.24_nominal.jsonl --output data/train_v8.25.jsonl \\
      --requests-file data/_review_semantic_role_train_requests.jsonl \\
      --api-key $ANTHROPIC_API_KEY --batch-id msgbatch_...
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

# ─── Taxonomie des rôles sémantiques (labels.py::SEMANTIC_ROLE_LABELS) ───────

SEMANTIC_ROLE_TAXONOMY = """## Rôles sémantiques possibles

- **AGENT** : initiateur de l'action, sujet actif ("le gouvernement annonce...")
- **PATIENT** : entité affectée / objet de l'action ("...annonce une réforme")
- **CONTENT** : contenu propositionnel — ce qui est dit/pensé/su ("il pense QUE...", objet d'un verbe de communication/cognition)
- **SOURCE** : source épistémique ("selon X", "d'après X", origine d'une information)
- **LOCATION** : lieu de l'action ou destination ("à Paris", "vers le nord")
- **TEMPORAL** : ancrage temporel ("en 2023", "pendant la crise", "hier")
- **CAUSE** : cause / déclencheur de l'événement ("à cause de X", "en raison de X")
- **PURPOSE** : but / intention ("pour X", "afin de X")
- **MEASURE** : quantité / valeur numérique ("de 3 millions", "d'environ 10 %")
- **BENEFICIARY** : bénéficiaire ("pour X", "en faveur de X", "au profit de X")
- **COMITATIVE** : co-participant ("avec X", "aux côtés de X", "accompagné de X")
- **ADVERSARY** : opposant ("contre X", "face à X")
- **DOMAIN** : domaine/thème abstrait concerné ("sur X", "en matière de X", "dans le domaine de X")
- **INSTRUMENT** : moyen/outil utilisé ("avec X", "via X", "à l'aide de X")
- **PART_OF** : appartenance à un tout ("fait partie de X", "membre de X", "de l'organisation X")
- **MEMBER_OF** : membre d'un ensemble/groupe
- **OWNER** : possesseur ("détenu par X", "propriété de X", "son X" = X possède)
- **IDENTITY** : apposition identitaire, X EST Y ("le PDG, Jean Dupont" -> Dupont=IDENTITY)
- **NONE** : aucun rôle sémantique pertinent (span hors argument, pas de relation claire)

Si AUCUN rôle ci-dessus n'est clairement applicable, réponds "OBLIQUE_UNRESOLVED"
(ne force pas un rôle si le contexte est trop ambigu — mieux vaut rester prudent)."""

SYSTEM_PROMPT = f"""Tu es un annotateur linguistique expert en français, spécialisé en rôles sémantiques (semantic role labeling).

{SEMANTIC_ROLE_TAXONOMY}

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un tableau JSON valide, aucun texte avant/après.
2. Pour chaque span numéroté [ROLE N], choisis EXACTEMENT un label dans la taxonomie
   ci-dessus (ou "OBLIQUE_UNRESOLVED" si indéterminable).
3. Base ta décision sur : la phrase complète, le hint NER du span, son svo_role
   (rôle syntaxique SUBJECT/OBJECT/OBLIQUE...), le verbe gouverneur (gov_verb_family)
   et/ou sa relation nominale (nominal_relation) si présents.
4. Ne modifie AUCUN autre champ (label NER, offsets, svo_role) — seulement le rôle sémantique.

## Format de réponse

```json
[
  {{"id": 1, "semantic_role": "DOMAIN", "raison": "complément thématique du nom"}},
  {{"id": 2, "semantic_role": "OBLIQUE_UNRESOLVED", "raison": "contexte trop ambigu"}}
]
```

Ne génère RIEN d'autre que le tableau JSON.
"""

VALID_ROLES = {
    "AGENT", "PATIENT", "CONTENT", "SOURCE", "LOCATION", "TEMPORAL", "CAUSE",
    "PURPOSE", "MEASURE", "BENEFICIARY", "COMITATIVE", "ADVERSARY", "DOMAIN",
    "INSTRUMENT", "PART_OF", "MEMBER_OF", "OWNER", "IDENTITY", "NONE",
    "OBLIQUE_UNRESOLVED",
}

TARGET_VALUE = "OBLIQUE_UNRESOLVED"  # ne réviser que les spans non résolus


# ─── Chargement / sélection des phrases à réviser ────────────────────────────

def load_phrases_to_review(input_path: Path) -> tuple[list[dict], list[dict]]:
    all_phrases = []
    to_review = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            all_phrases.append(ex)
            unresolved = [s for s in ex.get("spans", [])
                          if s.get("semantic_role") == TARGET_VALUE]
            if unresolved:
                to_review.append(ex)

    n_spans = sum(len([s for s in ex.get("spans", []) if s.get("semantic_role") == TARGET_VALUE])
                  for ex in to_review)
    print(f"📝 {len(all_phrases)} phrases au total | {len(to_review)} à réviser "
          f"({n_spans} spans {TARGET_VALUE})")
    return to_review, all_phrases


def build_user_prompt(batch: list[dict]) -> str:
    """IDs [ROLE N] globaux sur tout le batch (cohérent avec apply_corrections)."""
    parts = []
    global_id = 0
    for item in batch:
        spans = item.get("spans", [])
        unresolved = [(i, s) for i, s in enumerate(spans)
                      if s.get("semantic_role") == TARGET_VALUE]
        if not unresolved:
            continue

        text = item["text"]
        span_annotations = []
        for span_idx, s in unresolved:
            global_id += 1
            extra = []
            if s.get("svo_role"):        extra.append(f'svo_role={s["svo_role"]}')
            if s.get("gov_verb_family"): extra.append(f'gov_verb_family={s["gov_verb_family"]}')
            if s.get("nominal_relation"): extra.append(f'nominal_relation={s["nominal_relation"]}')
            extra_str = (", " + ", ".join(extra)) if extra else ""
            span_annotations.append(
                f'  [ROLE {global_id}] text="{s.get("text","")}", '
                f'label="{s.get("label","")}"{extra_str}'
            )

        parts.append(
            f'ID: {item.get("id","")}\n'
            f'Phrase: "{text}"\n'
            f'Spans à évaluer :\n' + "\n".join(span_annotations)
        )

    return (
        "Détermine le rôle sémantique des spans [ROLE] suivants. "
        "Les IDs sont GLOBAUX sur tout le batch.\n\n"
        + "\n\n---\n\n".join(parts)
    )


def parse_response(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return []


# ─── Batch API (pattern identique à review_stanza_spans_haiku_batch.py) ─────

def create_batch_requests(batches: list[list[dict]], output_jsonl: str, model: str):
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batches):
            user_prompt = build_user_prompt(batch)
            request = {
                "custom_id": f"batch_{batch_idx}",
                "params": {
                    "model": model,
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "system": [{"type": "text", "text": SYSTEM_PROMPT,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": user_prompt}],
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"📦 {len(batches)} requêtes batch → {output_jsonl}")


def submit_batch(api_key: str, requests_jsonl: str, max_retries: int = 6) -> str:
    url = "https://api.anthropic.com/v1/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    requests_list = []
    with open(requests_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests_list.append(json.loads(line))

    print(f"📤 Envoi de {len(requests_list)} requêtes...")
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json={"requests": requests_list})
                resp.raise_for_status()
                data = resp.json()
            batch_id = data["id"]
            id_file = requests_jsonl.replace(".jsonl", ".batch_id")
            Path(id_file).write_text(batch_id)
            print(f"✅ Batch créé : {batch_id}")
            return batch_id
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                wait = min(10 * 2 ** (attempt - 1), 120)
                print(f"⚠️  Tentative {attempt}/{max_retries} — HTTP {e.response.status_code}, retry dans {wait}s…")
                last_exc = e
                time.sleep(wait)
            else:
                raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = min(10 * 2 ** (attempt - 1), 120)
            print(f"⚠️  Tentative {attempt}/{max_retries} — réseau, retry dans {wait}s…")
            last_exc = e
            time.sleep(wait)
    raise RuntimeError(f"❌ submit_batch échoué après {max_retries} tentatives : {last_exc}")


def poll_batch(api_key: str, batch_id: str, poll_interval: int = 30) -> dict:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }
    with httpx.Client(timeout=60.0) as client:
        while True:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = data["processing_status"]
            counts = data.get("request_counts", {})
            print(f"⏳ Statut: {status} | {counts}")
            if status == "ended":
                return data
            time.sleep(poll_interval)


def fetch_batch_results(api_key: str, results_url: str) -> list[dict]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }
    results = []
    with httpx.Client(timeout=120.0) as client:
        with client.stream("GET", results_url, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    results.append(json.loads(line))
    return results


def apply_corrections(all_phrases: list[dict], to_review: list[dict],
                       results: list[dict], model: str) -> tuple[list[dict], Counter]:
    """Applique les verdicts sur les spans OBLIQUE_UNRESOLVED, dans le même
    ordre global que build_user_prompt (reconstruit les batches identiquement).
    """
    stats = Counter()

    # Reconstruire le mapping global_id -> (phrase_idx_in_to_review, span_idx)
    id_map = {}
    global_id = 0
    for phrase_idx, item in enumerate(to_review):
        spans = item.get("spans", [])
        unresolved = [(i, s) for i, s in enumerate(spans)
                      if s.get("semantic_role") == TARGET_VALUE]
        for span_idx, _ in unresolved:
            global_id += 1
            id_map[global_id] = (phrase_idx, span_idx)

    by_custom_id = {r["custom_id"]: r for r in results}
    review_by_id = {item.get("id"): item for item in to_review}

    for batch_idx in range(len([r for r in results])):
        custom_id = f"batch_{batch_idx}"
        r = by_custom_id.get(custom_id)
        if r is None:
            continue
        result_data = r.get("result", {})
        if result_data.get("type") != "succeeded":
            stats["batch_failed"] += 1
            continue
        message = result_data["message"]
        text = "".join(b["text"] for b in message["content"] if b["type"] == "text")
        verdicts = parse_response(text)

        for v in verdicts:
            gid = v.get("id")
            if gid not in id_map:
                stats["unknown_id"] += 1
                continue
            phrase_idx, span_idx = id_map[gid]
            new_role = v.get("semantic_role")
            if new_role not in VALID_ROLES:
                stats["invalid_role"] += 1
                continue
            span = to_review[phrase_idx]["spans"][span_idx]
            old_role = span.get("semantic_role")
            span["semantic_role"] = new_role
            span["_semantic_role_reviewed"] = True
            if new_role != old_role:
                stats["changed"] += 1
            else:
                stats["confirmed_unresolved"] += 1

    return all_phrases, stats


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--requests-file", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--batch-size", type=int, default=8, help="phrases par requête")
    ap.add_argument("--batch-id", default=None, help="reprendre un batch existant")
    ap.add_argument("--poll-interval", type=int, default=30)
    args = ap.parse_args()

    input_path = Path(args.input)
    to_review, all_phrases = load_phrases_to_review(input_path)

    if not to_review:
        print("✅ Rien à réviser — copie directe.")
        with open(args.output, "w", encoding="utf-8") as f:
            for row in all_phrases:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    batches = [to_review[i:i + args.batch_size] for i in range(0, len(to_review), args.batch_size)]

    if args.batch_id:
        batch_id = args.batch_id
        print(f"↪️  Reprise du batch existant : {batch_id}")
    else:
        create_batch_requests(batches, args.requests_file, args.model)
        batch_id = submit_batch(args.api_key, args.requests_file)

    batch_info = poll_batch(args.api_key, batch_id, args.poll_interval)
    results = fetch_batch_results(args.api_key, batch_info["results_url"])
    print(f"📥 {len(results)} résultats reçus")

    all_phrases, stats = apply_corrections(all_phrases, to_review, results, args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_phrases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("✅ RÉSULTAT")
    print("=" * 50)
    for k, v in stats.most_common():
        print(f"  {k:25s} : {v}")
    print(f"  Sortie : {args.output}")


if __name__ == "__main__":
    main()

