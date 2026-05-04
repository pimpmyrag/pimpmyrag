#!/usr/bin/env python3
"""
Relabélisation fine des spans hint_concept → 5 sous-types + fallback via Claude Haiku Batch API.

Lit un JSONL annoté (issu de preannotate_claude_batch.py, version v6.7+) et pour chaque span
portant le label hint_concept, demande à Haiku de le reclasser dans :
  - hint_rule         : règle, procédure, norme, protocole, standard
  - hint_doctrine     : doctrine, idéologie, courant de pensée, théorie (y compris nommée)
  - hint_state        : état, condition, situation abstraite (pauvreté, crise, guerre, paix…)
  - hint_notion       : notion, concept abstrait pur (liberté, justice, démocratie…)
  - hint_work_generic : production intellectuelle / média générique sans titre précis
  - hint_field        : domaine / secteur d'activité humaine (santé, éducation, agriculture…)
  - hint_process      : processus socio-économique continu (réforme, privatisation, transition…)
  - hint_concept      : fallback — concept inclassable dans les 7 sous-types ci-dessus

Note : depuis v6.7, hint_concept_named a été fusionné dans hint_concept
       (via data/remap_concept_named_to_concept.py). Ces spans nommés (théories,
       phénomènes) sont donc maintenant traités ici et orientés vers hint_doctrine.

Usage:
  python3 scripts/relabel_concepts_haiku_batch.py \\
    --input  data/train_wiki_claude_annotated.jsonl \\
    --output data/train_wiki_concepts_relabeled.jsonl \\
    --batch-size 20
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

# ─── Taxonomie des 5 sous-types de hint_concept ──────────────────────────────

CONCEPT_TAXONOMY = """## Relabélisation de hint_concept — 7 sous-types

Tu reçois des spans NER avec le label **hint_concept** et tu dois reclasser **chacun**
dans exactement l'un des 7 sous-types suivants.

### hint_rule
Règle, procédure, norme, protocole, standard, code de conduite, principe opérationnel.
Quelque chose qui prescrit un comportement ou une marche à suivre.
Exemples : "règlement", "procédure d'appel", "code de déontologie", "protocole sanitaire",
           "norme ISO", "règle du jeu", "principe de précaution"
→ NON si c'est un texte légal avec force normative → c'est hint_law (déjà annoté séparément).
→ NON si c'est une doctrine idéologique → hint_doctrine.

### hint_doctrine
Doctrine, idéologie, courant de pensée, théorie générale (nommée ou non),
système de croyances, philosophie politique/économique/religieuse collective.
Cela inclut aussi les théories et phénomènes portant un NOM PROPRE figé quand ils constituent
un système de pensée ou un cadre théorique reconnu.
Exemples génériques : "libéralisme", "marxisme", "keynésianisme", "populisme", "nationalisme",
                      "salafisme", "mercantilisme", "néolibéralisme", "souverainisme"
Exemples nommés    : "théorie de la relativité", "loi des grands nombres", "Big Data",
                     "effet Dunning-Kruger", "keynésianisme", "darwinisme"
→ NON si c'est une simple notion abstraite sans dimension idéologique/théorique → hint_notion.

### hint_state
État, condition, situation abstraite dans laquelle se trouvent des entités ou une société.
Quelque chose qui décrit une circonstance, un mode d'être, une phase.
Exemples : "pauvreté", "crise", "guerre", "paix", "chômage", "récession",
           "instabilité", "impunité", "corruption", "insécurité", "précarité"
→ Si c'est un événement discret (attentat, élection) → c'est hint_event_nominal (déjà annoté).
→ NON si c'est une notion purement abstraite → hint_notion.

### hint_notion
Notion, concept abstrait pur, valeur, principe philosophique ou méta-concept générique.
Un idéal, une abstraction ou un terme qui désigne une idée en soi, sans impliquer un état
de fait précis ni une règle prescriptive.
Exemples de valeurs    : "liberté", "justice", "démocratie", "laïcité", "solidarité",
                          "souveraineté", "dignité", "égalité", "vérité"
Exemples méta-concepts : "notion", "idée", "sens", "signification", "existence", "nature",
                          "vision", "influence", "mémoire", "conscience", "essence",
                          "intention", "principe", "concept", "pensée"
→ ⚠️ Un mot générique abstraitement utilisé comme concept ("notion", "idée", "sens") est
   hint_notion même s'il est très court — ne pas forcer hint_concept sur ces cas.
→ NON si c'est une règle opérationnelle → hint_rule.
→ NON si c'est un état concret de la société → hint_state.

### hint_work_generic
Production intellectuelle, culturelle ou médiatique désignée de façon GÉNÉRIQUE (sans titre précis).
Exemples médias    : "la presse", "les médias", "la littérature", "le cinéma", "la recherche",
                     "l'art contemporain", "les réseaux sociaux", "la télévision", "Internet"
Exemples génériques: "film" (sans titre), "œuvre" (sans titre), "livre" (sans titre),
                     "chanson", "album", "lettres", "morceaux musicaux", "classiques"
→ ⚠️ Un mot seul comme "film" ou "œuvre" utilisé sans titre précis = hint_work_generic.
→ Si c'est une œuvre avec un titre précis → c'est hint_work_of_art (déjà annoté séparément).
→ Si c'est un document non normatif concret → c'est hint_document (déjà annoté).

### hint_field
Domaine ou secteur d'activité humaine organisé : champ professionnel, disciplinaire ou économique.
Désigne un secteur dans lequel des acteurs exercent une activité, sans être un état ni une valeur.
Exemples : "l'agriculture", "la santé", "l'éducation", "l'énergie", "le tourisme",
           "l'industrie", "la défense", "le commerce", "les transports", "la finance",
           "l'environnement", "la culture", "le sport", "la diplomatie"
→ NON si c'est une valeur abstraite ("santé publique" comme principe) → hint_notion.
→ NON si c'est un processus de transformation → hint_process.

### hint_process
Processus socio-économique, politique ou technique CONTINU ou STRUCTUREL.
Différent d'un événement discret et ponctuel : c'est une dynamique de transformation qui s'étend
dans le temps et affecte une structure ou une société.
Exemples : "la réforme", "la privatisation", "la mondialisation", "la transition énergétique",
           "l'intégration européenne", "la désindustrialisation", "la restructuration",
           "la numérisation", "la démocratisation", "la radicalisation", "l'urbanisation"
→ NON si c'est un événement discret (arrestation, élection, attentat) → hint_event_nominal (déjà annoté).
→ NON si c'est un état stable résultant d'un processus → hint_state.

### hint_concept  ← fallback
Concept abstrait générique qui ne rentre VRAIMENT dans aucune des 7 catégories ci-dessus.
⚠️ ATTENTION : utilise ce label en DERNIER RECOURS seulement.
- Un terme qui exprime une valeur ou abstraction philosophique → hint_notion
- Un mot seul désignant une production culturelle → hint_work_generic
- Un domaine d'activité académique ou professionnel → hint_field
- Un processus de transformation continue → hint_process
Conserve hint_concept uniquement pour les termes qui restent vraiment inclassables
  même en regardant le contexte de la phrase.

## Règles importantes
1. Retourne le MÊME texte et les MÊMES offsets (start/end) — ne modifie PAS le span.
2. Essaie de reclasser chaque span dans l'un des 7 sous-types spécialisés.
3. Un terme court et générique ("notion", "film", "sens", "idée") a PRESQUE TOUJOURS
   un sous-type approprié — cherche-le avant de recourir à hint_concept.
4. Ne retourne "hint_concept" que pour les termes vraiment inclassables après avoir
   consulté le contexte de la phrase.
5. Si vraiment ambigu entre deux sous-types, choisis le plus probable selon le contexte.
"""

SYSTEM_PROMPT = f"""Tu es un expert en annotation sémantique pour le français.

{CONCEPT_TAXONOMY}

## Ta tâche
Pour chaque item, tu reçois :
- l'ID de la phrase
- le texte de la phrase
- une liste de spans hint_concept à reclasser

Retourne UNIQUEMENT un tableau JSON valide :
[
  {{
    "id": "...",
    "spans": [
      {{"start": int, "end": int, "text": "...", "label": "hint_rule|hint_doctrine|hint_state|hint_notion|hint_work_generic|hint_field|hint_process|hint_concept"}}
    ]
  }},
  ...
]

Chaque span doit conserver les mêmes start/end/text et n'avoir que le champ "label" modifié.
Utilise "hint_concept" si le span ne rentre clairement dans aucune des 7 catégories spécialisées.
Ne retourne RIEN d'autre que le JSON.
"""

# ─── Labels valides pour ce script ───────────────────────────────────────────

CONCEPT_SUBLABELS = {
    "hint_rule",
    "hint_doctrine",
    "hint_state",
    "hint_notion",
    "hint_work_generic",
    "hint_field",     # domaine / secteur d'activité
    "hint_process",   # processus continu de transformation
    "hint_concept",   # fallback explicite : conserve le label d'origine
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def repair_offset(text: str, span_text: str, hint_start: int, hint_end: int,
                  window: int = 60) -> tuple[int, int] | None:
    if not span_text:
        return None
    lo = max(0, hint_start - window)
    hi = min(len(text), hint_end + window)
    idx = text.find(span_text, lo, hi)
    if idx != -1:
        return idx, idx + len(span_text)
    idx = text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    return None


def build_user_prompt(batch: list[dict]) -> str:
    """
    batch : liste d'items {"id", "text", "concept_spans": [...]}
    concept_spans : spans hint_concept extraits de l'item
    """
    parts = []
    for item in batch:
        spans_payload = [
            {"start": sp["start"], "end": sp["end"], "text": sp["text"]}
            for sp in item["concept_spans"]
        ]
        parts.append(
            f'ID: {item["id"]}\n'
            f'Phrase: "{item["text"]}"\n'
            f'Spans hint_concept à reclasser: {json.dumps(spans_payload, ensure_ascii=False)}'
        )
    return "Reclasse chaque span hint_concept :\n\n" + "\n\n".join(parts)


def parse_response(response_text: str) -> list[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        results = json.loads(text)
        if isinstance(results, dict):
            if "annotations" in results:
                results = results["annotations"]
            else:
                results = [results]
        return results
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


# ─── Batch helpers ────────────────────────────────────────────────────────────

def _batch_id_file(requests_jsonl: str) -> str:
    return requests_jsonl.replace(".jsonl", ".batch_id")

def _save_batch_id(batch_id: str, requests_jsonl: str):
    path = _batch_id_file(requests_jsonl)
    with open(path, "w") as f:
        f.write(batch_id)
    print(f"💾 Batch ID sauvegardé → {path}")

def _load_batch_id(requests_jsonl: str) -> str | None:
    path = _batch_id_file(requests_jsonl)
    if os.path.exists(path):
        with open(path) as f:
            bid = f.read().strip()
        if bid:
            return bid
    return None


# ─── Étape 1 : Créer le fichier JSONL de requêtes ────────────────────────────

def create_batch_requests(batches: list[list[dict]], output_jsonl: str,
                          model: str = "claude-haiku-4-5"):
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batches):
            user_prompt = build_user_prompt(batch)
            request = {
                "custom_id": f"batch_{batch_idx}",
                "params": {
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "system": [{"type": "text", "text": SYSTEM_PROMPT,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": user_prompt}],
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"📦 {len(batches)} requêtes batch → {output_jsonl}")


# ─── Étape 2 : Soumettre ─────────────────────────────────────────────────────

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

    print(f"📤 Envoi de {len(requests_list)} requêtes au Batch API…")

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json={"requests": requests_list})
                resp.raise_for_status()
                data = resp.json()
            batch_id = data["id"]
            _save_batch_id(batch_id, requests_jsonl)
            print(f"✅ Batch créé : {batch_id}  |  Status: {data.get('processing_status', 'unknown')}")
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

    raise RuntimeError(
        f"❌ submit_batch échoué après {max_retries} tentatives. Dernière erreur : {last_exc}"
    )


# ─── Étape 3 : Poll ──────────────────────────────────────────────────────────

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
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            consecutive_errors += 1
            wait = min(15 * 2 ** (consecutive_errors - 1), 120)
            print(f"  ⚠️  Erreur poll, retry dans {wait}s… ({consecutive_errors} consécutives)")
            if consecutive_errors >= 8:
                raise RuntimeError(f"❌ poll_batch : trop d'erreurs consécutives — {e}") from e
            time.sleep(wait)
            continue

        status = data.get("processing_status", "unknown")
        counts = data.get("request_counts", {})
        elapsed = time.time() - t_start
        succeeded = counts.get("succeeded", 0)
        errored = counts.get("errored", 0)
        processing = counts.get("processing", 0)
        total = succeeded + errored + processing

        print(f"  ⏳ [{elapsed:.0f}s] {status} | ✅ {succeeded} | ❌ {errored} | 🔄 {processing}/{total}")
        if status == "ended":
            print(f"\n🎉 Batch terminé en {elapsed:.0f}s")
            return data
        time.sleep(poll_interval)


# ─── Étape 4 : Récupérer les résultats ───────────────────────────────────────

def fetch_results(api_key: str, batch_id: str, max_retries: int = 5) -> list[dict]:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    last_exc = None
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
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            wait = min(10 * 2 ** (attempt - 1), 120)
            print(f"⚠️  fetch_results tentative {attempt}/{max_retries} — retry dans {wait}s…")
            last_exc = e
            time.sleep(wait)
    raise RuntimeError(f"❌ fetch_results échoué après {max_retries} tentatives : {last_exc}")


# ─── Étape 5 : Appliquer le relabeling et écrire ─────────────────────────────

def process_results(results: list[dict], batches: list[list[dict]],
                    all_items_by_id: dict[str, dict], output: str):
    """
    Applique les nouveaux labels Haiku sur les spans hint_concept de chaque item,
    puis recopie tous les autres spans intacts.
    """
    batch_by_id = {f"batch_{i}": batch for i, batch in enumerate(batches)}

    label_stats = Counter()
    n_processed = 0
    n_errors = 0
    n_fallback = 0
    n_relabeled = 0
    n_unknown = 0

    with open(output, "a", encoding="utf-8") as out:
        for result in results:
            custom_id = result.get("custom_id", "")
            result_type = result.get("result", {}).get("type", "")
            batch = batch_by_id.get(custom_id, [])

            if result_type == "succeeded":
                message = result["result"]["message"]
                response_text = "".join(
                    b["text"] for b in message.get("content", []) if b.get("type") == "text"
                )
                parsed = parse_response(response_text)
                relabeled_by_id: dict[str, dict[int, str]] = {}  # id → {start → new_label}
                for item_result in parsed:
                    iid = item_result.get("id", "")
                    mapping = {}
                    for sp in item_result.get("spans", []):
                        new_label = sp.get("label", "")
                        if new_label in CONCEPT_SUBLABELS:
                            mapping[sp.get("start")] = new_label
                        else:
                            n_unknown += 1
                    relabeled_by_id[iid] = mapping

                for item_batch_entry in batch:
                    item_id = item_batch_entry["id"]
                    original = all_items_by_id[item_id]
                    mapping = relabeled_by_id.get(item_id, {})

                    new_spans = []
                    for sp in original.get("spans", []):
                        if sp.get("label") == "hint_concept":
                            if sp["start"] in mapping:
                                new_label = mapping[sp["start"]]
                                new_spans.append({**sp, "label": new_label})
                                label_stats[new_label] += 1
                                n_relabeled += 1
                            else:
                                # Haiku n'a pas répondu pour ce span → conserver hint_concept
                                new_spans.append(sp)  # label reste hint_concept
                                label_stats["hint_concept_no_response"] += 1
                                n_fallback += 1
                        else:
                            new_spans.append(sp)

                    record = {k: v for k, v in original.items() if k != "spans"}
                    record["spans"] = new_spans
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_processed += 1
            else:
                n_errors += 1
                error_msg = result.get("result", {}).get("error", {}).get("message", "unknown")
                print(f"  ❌ {custom_id}: {error_msg}")
                # Fallback : garder hint_concept tel quel
                for item_batch_entry in batch:
                    item_id = item_batch_entry["id"]
                    original = all_items_by_id[item_id]
                    out.write(json.dumps(original, ensure_ascii=False) + "\n")
                    n_processed += 1

    print(f"\n{'=' * 60}")
    print(f"✅ {n_processed} phrases écrites → {output}")
    print(f"🔄 {n_relabeled} spans hint_concept relabélisés")
    print(f"⚠️  {n_fallback} spans sans réponse Haiku → hint_concept conservé")
    print(f"❓ {n_unknown} labels inconnus ignorés")
    print(f"❌ {n_errors} batches en erreur")
    print(f"\n📊 Nouveaux labels :")
    for label, count in label_stats.most_common():
        print(f"  {label:<28} {count:>6}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Relabélise les spans hint_concept → 5 sous-types via Haiku Batch API"
    )
    parser.add_argument("--input", required=True,
                        help="JSONL d'entrée avec spans hint_concept")
    parser.add_argument("--output", required=True,
                        help="JSONL de sortie avec spans relabélisés")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Nombre de phrases (items) par requête batch")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--model", default="claude-haiku-4-5",
                        help="Modèle à utiliser (par défaut : claude-haiku-4-5)")
    parser.add_argument("--batch-id", default=None,
                        help="Reprendre un batch Anthropic existant")
    parser.add_argument("--requests-file", default="data/_haiku_concept_requests.jsonl",
                        help="Chemin du JSONL de requêtes à créer/réutiliser")
    parser.add_argument("--results-file", default=None,
                        help="Fichier de résultats JSONL déjà téléchargé (skip submit+poll+fetch)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not args.results_file and not api_key:
        print("❌ Clé API manquante. --api-key ou ANTHROPIC_API_KEY")
        sys.exit(1)

    # ── Charger l'input ────────────────────────────────────────────
    all_items: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            all_items.append(json.loads(line))
            if args.max_sentences and len(all_items) >= args.max_sentences:
                break
    print(f"📝 {len(all_items)} phrases chargées")

    # Index par ID pour lookup rapide
    all_items_by_id: dict[str, dict] = {item["id"]: item for item in all_items}

    # ── Vérifier les déjà traités ──────────────────────────────────
    already_done: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        already_done.add(json.loads(line).get("id", ""))
                    except json.JSONDecodeError:
                        pass
        if already_done:
            print(f"🔄 {len(already_done)} déjà traitées, skip")

    # ── Filtrer les items qui ont au moins un hint_concept ─────────
    candidates = [
        item for item in all_items
        if item["id"] not in already_done
        and any(sp.get("label") == "hint_concept" for sp in item.get("spans", []))
    ]
    print(f"🔍 {len(candidates)} phrases contenant hint_concept à relabéliser")

    # Items sans hint_concept → recopier directement
    no_concept = [
        item for item in all_items
        if item["id"] not in already_done
        and not any(sp.get("label") == "hint_concept" for sp in item.get("spans", []))
    ]
    if no_concept:
        print(f"⏭️  {len(no_concept)} phrases sans hint_concept → recopie directe")
        with open(args.output, "a", encoding="utf-8") as out:
            for item in no_concept:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not candidates:
        print("✅ Rien à relabéliser !")
        return

    # Préparer les batches : on enrichit chaque item avec concept_spans
    batch_items = []
    for item in candidates:
        concept_spans = [sp for sp in item.get("spans", []) if sp.get("label") == "hint_concept"]
        batch_items.append({
            "id": item["id"],
            "text": item["text"],
            "concept_spans": concept_spans,
        })

    batches: list[list[dict]] = []
    for i in range(0, len(batch_items), args.batch_size):
        batches.append(batch_items[i: i + args.batch_size])

    # ── Mode résultats locaux ──────────────────────────────────────
    if args.results_file:
        print(f"📂 Résultats locaux depuis {args.results_file}…")
        results = []
        with open(args.results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"📥 {len(results)} résultats chargés")
        process_results(results, batches, all_items_by_id, args.output)
        return

    # ── Submit / reprise ──────────────────────────────────────────
    if not args.batch_id:
        saved_id = _load_batch_id(args.requests_file)
        if saved_id:
            print(f"🔄 Batch ID trouvé en cache : {saved_id}")
            batch_id = saved_id
        else:
            create_batch_requests(batches, args.requests_file, args.model)
            batch_id = submit_batch(api_key, args.requests_file)
    else:
        batch_id = args.batch_id
        print(f"🔄 Reprise batch {batch_id}")

    poll_batch(api_key, batch_id, args.poll_interval)
    results = fetch_results(api_key, batch_id)
    process_results(results, batches, all_items_by_id, args.output)


if __name__ == "__main__":
    main()

