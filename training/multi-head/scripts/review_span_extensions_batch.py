#!/usr/bin/env python3
"""
Validation des extensions de spans NP via Claude Haiku Batch API.

Pour chaque phrase du fichier candidates_span_extension_v8.17_clean.jsonl,
Claude valide si chaque extension proposée est sémantiquement correcte
(le span étendu désigne-t-il bien la même entité avec plus de précision ?).

Verdict par candidat :
  true  → extension acceptée → le span sera élargi
  false → extension refusée → span inchangé

Usage :
  python3 scripts/review_span_extensions_batch.py \\
    --candidates data/candidates_span_extension_v8.17_clean.jsonl \\
    --train-input  data/train_v8.17.jsonl \\
    --val-input    data/val_v8.17.jsonl \\
    --test-input   data/test_v8.17.jsonl \\
    --train-output data/train_v8.18.jsonl \\
    --val-output   data/val_v8.18.jsonl \\
    --test-output  data/test_v8.18.jsonl \\
    --requests-file data/_extend_spans_requests.jsonl \\
    --api-key $ANTHROPIC_API_KEY \\
    --poll-interval 30

  # Reprendre un batch existant :
    --batch-id msgbatch_...
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

# ─── Prompt système ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en annotation NER pour le français.
On t'a proposé des extensions de spans : au lieu d'annoter seulement le mot-tête
("président"), on voudrait annoter le syntagme nominal complet ("ancien président",
"président de la République", "président élu").

Règles d'acceptation :
✅ ACCEPTER si l'extension ajoute des modificateurs cohérents avec le label :
   - Adjectifs qualificatifs ("industrielle", "sanitaire", "structurelle")
   - Adjectifs antéposés ("nouveau", "ancien", "premier", "jeune")
   - Compléments du nom courts et cohérents ("de la République", "de maintenance")
   - Noms composés ("cessez-le-feu", "centre-ville", "nord-est", "enseignants-chercheurs")
   - Numéros ou quantificateurs collant au nom ("deux meurtres", "40 agents")

❌ REFUSER si :
   - L'extension inclut des mots qui appartiennent à une autre entité
   - L'extension change le sens (désigne une entité différente)
   - L'extension est syntaxiquement incohérente avec le label
   - Le texte étendu commence ou se termine sur un mot de liaison parasite
   - L'extension est clairement un artefact (liste, tableau, répétition)

Réponds UNIQUEMENT en JSON valide, sans texte avant ni après :
[{"n": 1, "ok": true}, {"n": 2, "ok": false}, ...]"""


# ─── Construction des requêtes ────────────────────────────────────────────────

def build_request(rec: dict) -> dict:
    """
    Construit une requête Claude pour une phrase avec ses candidats.
    custom_id = "{split}__{id}" pour garantir l'unicité même si un id
    apparaît dans plusieurs splits.
    """
    split   = rec["split"]
    sent_id = rec["id"]
    text    = rec["text"]
    spans   = rec.get("spans", [])
    cands   = rec.get("candidates", [])

    # Contexte NER existant (spans hint_ seulement, pour aider Claude)
    hint_spans = [s for s in spans if s.get("label", "").startswith("hint_")]
    context_parts = [f'"{s["text"]}" ({s["label"]})' for s in hint_spans[:10]]
    context_str = ", ".join(context_parts) if context_parts else "(aucun)"

    # Liste des extensions proposées
    ext_lines = []
    for n, c in enumerate(cands, 1):
        ext_lines.append(
            f'[{n}] label={c["label"]} | '
            f'actuel: "{c["current_text"]}" ({c["current_start"]}:{c["current_end"]}) → '
            f'étendu: "{c["candidate_text"]}" ({c["candidate_start"]}:{c["candidate_end"]})'
        )

    user_prompt = (
        f'Phrase : "{text}"\n'
        f'NER existant : {context_str}\n\n'
        f'Extensions à valider :\n' + "\n".join(ext_lines)
    )

    custom_id = f"{split}__{sent_id}"

    return {
        "custom_id": custom_id,
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "temperature": 0.0,
            "system": [{"type": "text", "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_prompt}],
        }
    }


def create_requests_file(candidates_path: str, output_jsonl: str):
    records = []
    with open(candidates_path) as f:
        for line in f:
            records.append(json.loads(line))

    # custom_id court (max 64 chars) : {split_prefix}{index:05d}
    # ex: tr00001, va00001, te00001
    split_prefix = {"train": "tr", "val": "va", "test": "te"}
    split_counters = Counter()

    # Mapping custom_id -> (split, sentence_id) sauvegardé à côté
    id_map = {}   # custom_id -> {"split": ..., "id": ...}

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            split = rec["split"]
            split_counters[split] += 1
            prefix = split_prefix.get(split, split[:2])
            custom_id = f"{prefix}{split_counters[split]:05d}"   # ex: tr00001 (8 chars max)
            id_map[custom_id] = {"split": split, "id": rec["id"]}

            req = build_request(rec)
            req["custom_id"] = custom_id
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # Sauvegarder le mapping
    map_path = output_jsonl.replace(".jsonl", "_id_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=None)

    print(f"📦 {sum(split_counters.values())} requêtes → {output_jsonl}")
    print(f"🗺️  Mapping IDs → {map_path}")
    for s, n in sorted(split_counters.items()):
        print(f"   {s:<8}: {n:>5} phrases")

    # Vérifier max longueur custom_id
    max_len = max(len(k) for k in id_map)
    print(f"   custom_id max len: {max_len} (limite API: 64)")


# ─── Batch API ────────────────────────────────────────────────────────────────

def submit_batch(api_key: str, requests_jsonl: str, max_retries: int = 6) -> str:
    url = "https://api.anthropic.com/v1/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    requests_list = []
    with open(requests_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests_list.append(json.loads(line))

    print(f"📤 Envoi de {len(requests_list)} requêtes...")
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Test préalable avec 1 requête pour vérifier le format
            if attempt == 1:
                print("🔍 Test avec 1 requête...")
                with httpx.Client(timeout=30.0) as client:
                    r = client.post(url, headers=headers, json={"requests": requests_list[:1]})
                    if not r.is_success:
                        print(f"❌ Test échoué HTTP {r.status_code} : {r.text[:500]}")
                        sys.exit(1)
                    test_id = r.json().get("id", "?")
                    print(f"✅ Test OK (batch_id préliminaire: {test_id}) — envoi complet...")
                    # Annuler le batch de test immédiatement
                    try:
                        client.post(f"https://api.anthropic.com/v1/messages/batches/{test_id}/cancel", headers=headers)
                    except Exception:
                        pass

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json={"requests": requests_list})
                if not resp.is_success:
                    print(f"❌ HTTP {resp.status_code} : {resp.text[:500]}")
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
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    t_start = time.time()
    consecutive_errors = 0
    while True:
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            wait = min(15 * 2 ** (consecutive_errors - 1), 120)
            print(f"  ⚠️  Erreur poll ({type(e).__name__}), retry dans {wait}s…")
            if consecutive_errors >= 8:
                raise RuntimeError(f"❌ poll_batch : trop d'erreurs — {e}") from e
            time.sleep(wait)
            continue

        status = data.get("processing_status", "unknown")
        counts = data.get("request_counts", {})
        elapsed = time.time() - t_start
        succeeded  = counts.get("succeeded", 0)
        errored    = counts.get("errored", 0)
        processing = counts.get("processing", 0)
        total = succeeded + errored + processing
        print(f"  ⏳ [{elapsed:.0f}s] {status} | ✅ {succeeded} | ❌ {errored} | 🔄 {processing}/{total}")
        if status == "ended":
            print(f"🎉 Batch terminé en {elapsed:.0f}s")
            return data
        time.sleep(poll_interval)


def fetch_results(api_key: str, batch_id: str, max_retries: int = 5) -> list[dict]:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    for attempt in range(1, max_retries + 1):
        results = []
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    for chunk in resp.iter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                try:
                                    results.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
            print(f"📥 {len(results)} résultats récupérés")
            return results
        except Exception as e:
            wait = min(10 * 2 ** (attempt - 1), 120)
            print(f"⚠️  fetch_results tentative {attempt}/{max_retries} — {type(e).__name__}, retry dans {wait}s…")
            time.sleep(wait)
    raise RuntimeError("❌ fetch_results échoué")


# ─── Parsing réponse Claude ───────────────────────────────────────────────────

def parse_response(text: str) -> list[dict]:
    """Parse la réponse JSON de Claude : [{"n": 1, "ok": true}, ...]"""
    text = text.strip()
    # Retire les backticks éventuels
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


# ─── Application des extensions ──────────────────────────────────────────────

def apply_extensions(
    candidates_path: str,
    train_input: str, val_input: str, test_input: str,
    train_output: str, val_output: str, test_output: str,
    results: list[dict],
    requests_file: str,
):
    """
    Applique les extensions acceptées par Claude sur les datasets v8.17.
    Utilise le fichier de mapping custom_id → (split, sentence_id).
    """

    # ── 1. Charger le mapping custom_id → (split, sentence_id) ─────────────
    map_path = requests_file.replace(".jsonl", "_id_map.json")
    with open(map_path) as f:
        id_map = json.load(f)   # custom_id -> {"split": ..., "id": ...}
    print(f"🗺️  Mapping chargé : {len(id_map)} entrées")

    # ── 2. Charger les candidats indexés par (split, sentence_id) ───────────
    cands_by_key: dict[tuple, list[dict]] = {}
    with open(candidates_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["split"], rec["id"])
            cands_by_key[key] = rec["candidates"]

    # ── 3. Construire le mapping custom_id → extensions acceptées ───────────
    accepted: dict[tuple, list[dict]] = {}   # (split, sentence_id) → candidats acceptés
    stats = Counter()

    for result in results:
        custom_id   = result.get("custom_id", "")
        result_type = result.get("result", {}).get("type", "")

        if result_type != "succeeded":
            stats["errored"] += 1
            continue

        response_text = "".join(
            b["text"] for b in result["result"]["message"].get("content", [])
            if b.get("type") == "text"
        )
        verdicts = parse_response(response_text)
        if not verdicts:
            stats["parse_failed"] += 1
            continue

        # Récupérer (split, sentence_id) depuis le mapping
        info = id_map.get(custom_id)
        if not info:
            stats["unknown_id"] += 1
            continue

        key = (info["split"], info["id"])
        cands = cands_by_key.get(key, [])

        # Construire dict n → ok
        verdict_map = {}
        for v in verdicts:
            n  = v.get("n") or v.get("id") or v.get("idx")
            ok = v.get("ok") or v.get("accept") or v.get("accepted")
            if n is not None:
                verdict_map[int(n)] = bool(ok)

        accepted_for_sent = []
        for idx, c in enumerate(cands, 1):
            if verdict_map.get(idx, False):
                accepted_for_sent.append(c)
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1

        accepted[key] = accepted_for_sent
        stats["sentences_processed"] += 1

    print(f"\n📊 Résultats Claude :")
    for k, v in sorted(stats.items()):
        print(f"   {k:<25}: {v}")

    # ── 4. Appliquer sur chaque split ────────────────────────────────────────
    split_map = {
        "train": (train_input, train_output),
        "val":   (val_input,   val_output),
        "test":  (test_input,  test_output),
    }

    total_extended = 0
    for split, (inp, out) in split_map.items():
        n_extended = 0
        with open(inp) as fin, open(out, "w") as fout:
            for line in fin:
                rec = json.loads(line)
                key = (split, rec["id"])
                ext_list = accepted.get(key, [])

                if ext_list:
                    # Index des extensions par (current_start, current_end, label)
                    ext_index: dict[tuple, dict] = {}
                    for c in ext_list:
                        k = (c["current_start"], c["current_end"], c["label"])
                        if k not in ext_index or len(c["candidate_text"]) > len(ext_index[k]["candidate_text"]):
                            ext_index[k] = c

                    # Positions déjà occupées dans le jeu de spans original
                    # (candidate_start, candidate_end, label) → span existant
                    existing_positions: dict[tuple, dict] = {}
                    for s in rec.get("spans", []):
                        pk = (s.get("start"), s.get("end"), s.get("label"))
                        existing_positions[pk] = s

                    spans_to_drop = set()   # (start, end, label) des spans courts à supprimer
                    spans_to_extend = {}    # (start, end, label) → c  des spans à étendre

                    for c in ext_list:
                        short_k = (c["current_start"], c["current_end"], c["label"])
                        long_k  = (c["candidate_start"], c["candidate_end"], c["label"])

                        if long_k in existing_positions:
                            # La position cible existe déjà → supprimer le span court,
                            # pas besoin d'étendre (évite doublons et perte de SVO)
                            spans_to_drop.add(short_k)
                        else:
                            spans_to_extend[short_k] = c

                    new_spans = []
                    for s in rec.get("spans", []):
                        k = (s.get("start"), s.get("end"), s.get("label"))
                        if k in spans_to_drop:
                            n_extended += 1   # compté quand même (le long span couvre déjà)
                            continue          # supprime le doublon court
                        if k in spans_to_extend:
                            c = spans_to_extend[k]
                            s = dict(s)
                            s["start"] = c["candidate_start"]
                            s["end"]   = c["candidate_end"]
                            s["text"]  = c["candidate_text"]
                            n_extended += 1
                        new_spans.append(s)
                    rec["spans"] = new_spans

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"   {split:<8}: {n_extended:>5} spans étendus → {out}")
        total_extended += n_extended

    print(f"\n✅ Total spans étendus : {total_extended}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Span extension review via Claude Haiku Batch")
    p.add_argument("--candidates",    required=True,  help="candidates_span_extension_v8.17_clean.jsonl")
    p.add_argument("--train-input",   required=True)
    p.add_argument("--val-input",     required=True)
    p.add_argument("--test-input",    required=True)
    p.add_argument("--train-output",  required=True)
    p.add_argument("--val-output",    required=True)
    p.add_argument("--test-output",   required=True)
    p.add_argument("--requests-file", required=True,  help="Fichier JSONL des requêtes batch")
    p.add_argument("--api-key",       default=os.environ.get("ANTHROPIC_API_KEY", ""))
    p.add_argument("--batch-id",      default=None,   help="Reprendre un batch existant")
    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--dry-run",       action="store_true", help="Génère les requêtes sans envoyer")
    return p.parse_args()


def main():
    args = parse_args()

    DATA_DIR = Path(args.candidates).parent

    # Charger .secrets.env si dispo
    secrets_path = Path(__file__).parent.parent / ".secrets.env"
    if secrets_path.exists():
        from dotenv import load_dotenv
        load_dotenv(str(secrets_path))
    if not args.api_key:
        args.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Phase 1 : Construire les requêtes ────────────────────────────────────
    if not args.batch_id:
        print("🔧 Génération des requêtes batch...")
        create_requests_file(args.candidates, args.requests_file)

        if args.dry_run:
            print("✋ --dry-run : arrêt avant envoi")
            return

        if not args.api_key:
            print("❌ ANTHROPIC_API_KEY manquant")
            sys.exit(1)

        # ── Phase 2 : Soumettre ──────────────────────────────────────────────
        batch_id = submit_batch(args.api_key, args.requests_file)

    else:
        batch_id = args.batch_id
        print(f"🔄 Reprise du batch : {batch_id}")

    # ── Phase 3 : Attendre ───────────────────────────────────────────────────
    poll_batch(args.api_key, batch_id, poll_interval=args.poll_interval)

    # ── Phase 4 : Récupérer et appliquer ────────────────────────────────────
    results = fetch_results(args.api_key, batch_id)

    # Sauvegarder les résultats bruts
    raw_results_path = args.requests_file.replace(".jsonl", f"_{batch_id}_results.jsonl")
    with open(raw_results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Résultats bruts sauvegardés : {raw_results_path}")

    apply_extensions(
        candidates_path=args.candidates,
        train_input=args.train_input,   val_input=args.val_input,   test_input=args.test_input,
        train_output=args.train_output, val_output=args.val_output, test_output=args.test_output,
        results=results,
        requests_file=args.requests_file,
    )


if __name__ == "__main__":
    main()

