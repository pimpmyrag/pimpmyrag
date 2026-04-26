#!/usr/bin/env python3
"""
Pré-annotation NER via Claude Batch API.
Soumet toutes les requêtes en batch, poll le résultat, parse et écrit.

Usage:
  python3 scripts/preannotate_claude_batch.py \
    --input data/train_wiki_svo_ner.jsonl \
    --output data/train_wiki_claude_annotated.jsonl \
    --batch-size 5
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

# ─── Taxonomie et prompt (identique à la version Mistral) ───

TAXONOMY = """## Taxonomie NER — 31 labels fins groupés en 8 catégories

### PER (Personnes)
- **hint_person_name** : nom propre de personne ("Emmanuel Macron", "Jean Dupont")
- **hint_person_role** : rôle, titre, fonction ("président", "ministre", "médecin") — SANS le nom propre
- **hint_norp** : nationalité, ethnie, religion, parti politique ("français", "catholiques", "républicains")
- **hint_group_role** : groupe de personnes par rôle ("les soldats", "les manifestants", "la police")

### LOC (Lieux)
- **hint_gpe** : entité géopolitique — pays, ville, région ("France", "Paris", "Bretagne")
- **hint_fac_name** : bâtiment, monument, installation nommée ("Tour Eiffel", "gare de Lyon", "aéroport Charles-de-Gaulle")
- **hint_loc_generic** : lieu générique non-GPE ("la frontière", "la côte", "le centre-ville")
- **hint_infra** : infrastructure ("autoroute", "pont", "voie ferrée", "pipeline")

### ORG (Organisations)
- **hint_org_name** : nom d'organisation ("ONU", "Apple", "Médecins sans frontières")

### TIME (Temps)
- **hint_time_date** : date, jour, année ("14 juillet 2024", "lundi", "2023")
- **hint_time_clock** : heure précise ("13h00", "midi", "18 h 46")
- **hint_time_duration** : durée ("trois jours", "depuis 2020", "pendant six mois")

### EVENT (Événements)
- **hint_event_nominal** : événement décrit par un nom commun ("élection", "attentat", "crise", "manifestation")
- **hint_event_named** : événement nommé ("Jeux olympiques de Paris 2024", "COP28", "guerre du Vietnam")

### OBJECT (Objets)
- **hint_weapon** : arme ("fusil", "missile", "couteau", "bombe")
- **hint_vehicle** : véhicule ("avion", "voiture", "navire", "TGV")
- **hint_substance** : substance, matière première ("pétrole", "uranium", "chlore", "eau")
- **hint_food** : aliment, boisson ("blé", "vin", "fromage", "café")
- **hint_tool** : outil, instrument, appareil ("radar", "téléphone", "scanner")
- **hint_object_generic** : objet physique autre ("drapeau", "colis", "document")
- **hint_object_name** : objet nommé / marque ("iPhone", "Rafale", "Falcon 9")

### VALUE (Valeurs)
- **hint_measure** : mesure physique avec unité ("35 nœuds", "52 000 m³", "37°C", "120 dB", "15 km")
- **hint_percentage** : pourcentage ("45 %", "un tiers")
- **hint_count** : nombre de choses ou personnes ("trois personnes", "200 soldats", "six buts")
- **hint_money** : montant monétaire ("15 millions d'euros", "2,5 milliards de dollars")
- **hint_rate** : taux, ratio ("3,5 %", "1 pour 1000")

### ABSTRACT (Abstraits)
- **hint_law** : loi, traité, texte juridique ("RGPD", "loi Climat", "article 49.3", "Constitution")
- **hint_work_of_art** : œuvre d'art, livre, film ("Les Misérables", "La Joconde")
- **hint_concept** : concept abstrait ("démocratie", "laïcité", "souveraineté")
- **hint_disease** : maladie, pathologie ("Covid-19", "grippe aviaire", "cancer")
- **hint_language** : langue ("français", "mandarin", "arabe")

## Règles importantes
1. Les spans ne doivent PAS se chevaucher (mais l'imbrication est autorisée si types différents)
2. "le président Macron" → annoter SÉPARÉMENT : "président" = hint_person_role, "Macron" = hint_person_name
3. Les déterminants (le, la, les, un, une) ne font PAS partie du span sauf s'ils sont indissociables du nom
4. Annoter TOUTES les entités de la phrase, pas seulement les labels rares
5. Le champ "start" est l'index du premier caractère du span, "end" est l'index APRÈS le dernier caractère
"""

EXAMPLES = """## Exemples corrects

Phrase: "Les accords de Matignon ont mis fin à la crise en Nouvelle-Calédonie."
Spans: [{"label": "hint_law", "start": 4, "end": 23, "text": "accords de Matignon"}, {"label": "hint_gpe", "start": 50, "end": 68, "text": "Nouvelle-Calédonie"}]

Phrase: "L'oromo est la langue la plus parlée d'Éthiopie."
Spans: [{"label": "hint_language", "start": 2, "end": 7, "text": "oromo"}, {"label": "hint_gpe", "start": 39, "end": 47, "text": "Éthiopie"}]

Phrase: "La loi Climat et Résilience de 2021 fixe des objectifs ambitieux de réduction des émissions."
Spans: [{"label": "hint_law", "start": 3, "end": 27, "text": "loi Climat et Résilience"}, {"label": "hint_time_date", "start": 31, "end": 35, "text": "2021"}]

Phrase: "Plusieurs radars ont été neutralisés notamment à Saint-Pierre (59), Calais et Talmont-Saint-Hilaire."
Spans: [{"label": "hint_tool", "start": 10, "end": 16, "text": "radars"}, {"label": "hint_gpe", "start": 49, "end": 61, "text": "Saint-Pierre"}, {"label": "hint_gpe", "start": 68, "end": 74, "text": "Calais"}, {"label": "hint_gpe", "start": 78, "end": 99, "text": "Talmont-Saint-Hilaire"}]

Phrase: "Nietzsche a proclamé la mort de Dieu dans Ainsi parlait Zarathoustra."
Spans: [{"label": "hint_person_name", "start": 0, "end": 9, "text": "Nietzsche"}, {"label": "hint_work_of_art", "start": 42, "end": 68, "text": "Ainsi parlait Zarathoustra"}]
"""

SYSTEM_PROMPT = f"""Tu es un expert en annotation NER (Named Entity Recognition) pour le français.

{TAXONOMY}

{EXAMPLES}

## Ta tâche
Pour chaque phrase, tu reçois des pré-annotations (spans + labels) produites par un modèle NER automatique.
Tu dois :
1. CORRIGER les labels erronés
2. CORRIGER les frontières de spans (trop larges ou trop étroites)
3. AJOUTER les entités manquées par le modèle
4. SUPPRIMER les faux positifs évidents

Retourne UNIQUEMENT un JSON valide, un objet par phrase, dans un tableau JSON.
Chaque objet a : {{"id": "...", "text": "...", "spans": [...]}}
Chaque span a : {{"label": "...", "start": int, "end": int, "text": "..."}}

IMPORTANT : "start" et "end" doivent correspondre EXACTEMENT à text[start:end] == span["text"].
Ne retourne RIEN d'autre que le JSON.
"""

def repair_offset(text: str, span_text: str, hint_start: int, hint_end: int,
                  window: int = 60) -> tuple[int, int] | None:
    """
    Tente de retrouver la bonne position de span_text dans text.
    1. Cherche dans une fenêtre autour de l'offset suggéré par Claude.
    2. Si non trouvé, cherche dans tout le texte (première occurrence).
    3. Retourne (start, end) corrigés, ou None si introuvable.
    """
    if not span_text:
        return None
    # Fenêtre locale
    lo = max(0, hint_start - window)
    hi = min(len(text), hint_end + window)
    idx = text.find(span_text, lo, hi)
    if idx != -1:
        return idx, idx + len(span_text)
    # Recherche globale
    idx = text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    return None


VALID_LABELS = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_infra", "hint_weapon", "hint_vehicle", "hint_substance",
    "hint_food", "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named", "hint_time_date",
    "hint_time_clock", "hint_time_duration", "hint_measure",
    "hint_percentage", "hint_count", "hint_money", "hint_rate",
    "hint_law", "hint_work_of_art", "hint_concept", "hint_disease", "hint_language",
}


def build_user_prompt(batch: list[dict]) -> str:
    """
    Construit le prompt utilisateur à partir des items.
    Chaque item a un champ "spans" contenant les spans hint_* prédits par le modèle
    (marqués _predicted=True) ainsi que les spans svo_*/pron_* (ignorés ici).
    """
    parts = []
    for item in batch:
        # Extraire uniquement les spans hint_* prédits par le modèle
        preds = [
            {"label": sp["label"], "start": sp["start"], "end": sp["end"], "text": sp["text"]}
            for sp in item.get("spans", [])
            if sp.get("label", "").startswith("hint_")
        ]
        parts.append(
            f'ID: {item["id"]}\n'
            f'Phrase: "{item["text"]}"\n'
            f'Pré-annotations: {json.dumps(preds, ensure_ascii=False)}'
        )
    return "Corrige les annotations suivantes :\n\n" + "\n\n".join(parts)


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
            # Peut être {"annotations": [...]} ou directement un record
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


# ─── Étape 1 : Créer le fichier JSONL de requêtes pour le Batch API ───

def create_batch_requests(candidates: list[dict], batch_size: int, output_jsonl: str, args_model: str = "claude-sonnet-4-6"):
    """Crée le fichier JSONL des requêtes batch Claude."""
    batches = []
    for i in range(0, len(candidates), batch_size):
        batches.append(candidates[i : i + batch_size])

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batches):
            user_prompt = build_user_prompt(batch)
            # Format requis par Claude Batch API
            request = {
                "custom_id": f"batch_{batch_idx}",
                "params": {
                    "model": args_model,
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"📦 {len(batches)} requêtes batch → {output_jsonl}")
    return batches


# ─── Étape 2 : Soumettre le batch ───

def submit_batch(api_key: str, requests_jsonl: str) -> str:
    """Soumet le batch à Claude et retourne le batch_id."""
    url = "https://api.anthropic.com/v1/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "message-batches-2024-09-24",
    }

    # Lire le fichier JSONL et construire les requests
    requests = []
    with open(requests_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests.append(json.loads(line))

    payload = {"requests": requests}

    print(f"📤 Envoi de {len(requests)} requêtes au Batch API...")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    batch_id = data["id"]
    print(f"✅ Batch créé : {batch_id}")
    print(f"   Status: {data.get('processing_status', 'unknown')}")
    return batch_id


# ─── Étape 3 : Poll le status ───

def poll_batch(api_key: str, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll le batch jusqu'à completion."""
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }

    t_start = time.time()
    while True:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        status = data.get("processing_status", "unknown")
        counts = data.get("request_counts", {})
        elapsed = time.time() - t_start

        succeeded = counts.get("succeeded", 0)
        errored = counts.get("errored", 0)
        processing = counts.get("processing", 0)
        total = succeeded + errored + processing

        print(f"  ⏳ [{elapsed:.0f}s] Status: {status} | "
              f"✅ {succeeded} | ❌ {errored} | 🔄 {processing} / {total}")

        if status == "ended":
            print(f"\n🎉 Batch terminé en {elapsed:.0f}s")
            return data

        time.sleep(poll_interval)


# ─── Étape 4 : Récupérer les résultats ───

def fetch_results(api_key: str, batch_id: str) -> list[dict]:
    """Récupère les résultats du batch."""
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }

    results = []
    with httpx.Client(timeout=120.0) as client:
        # Stream les résultats (JSONL)
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


# ─── Étape 5 : Parser et écrire ───

def process_results(results: list[dict], batches: list[list[dict]], output: str):
    """
    Parse les résultats batch et écrit le JSONL final.
    Les spans svo_*/pron_* de l'input sont PRÉSERVÉS dans la sortie,
    seules les annotations hint_* sont remplacées par celles de Claude.
    """
    batch_by_id = {f"batch_{i}": batch for i, batch in enumerate(batches)}

    label_stats = Counter()
    n_processed = 0
    n_errors = 0
    n_fallback = 0

    with open(output, "w", encoding="utf-8") as out:
        for result in results:
            custom_id = result.get("custom_id", "")
            result_type = result.get("result", {}).get("type", "")
            batch = batch_by_id.get(custom_id, [])

            if result_type == "succeeded":
                message = result["result"]["message"]
                content_blocks = message.get("content", [])
                response_text = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        response_text += block["text"]

                parsed = parse_response(response_text)
                result_by_id = {r["id"]: r for r in parsed if "id" in r}

                for item in batch:
                    item_id = item["id"]
                    # Spans non-NER (svo_*, pron_*) à conserver tels quels
                    svo_spans = [
                        sp for sp in item.get("spans", [])
                        if not sp.get("label", "").startswith("hint_")
                    ]

                    if item_id in result_by_id:
                        corrected = result_by_id[item_id]
                        ner_spans = []
                        for s in corrected.get("spans", []):
                            if all(k in s for k in ("label", "start", "end", "text")):
                                if s["label"] in VALID_LABELS:
                                    span_text = s["text"]
                                    start, end = s["start"], s["end"]
                                    if item["text"][start:end] == span_text:
                                        ner_spans.append(s)
                                        label_stats[s["label"]] += 1
                                    else:
                                        # Tentative de réparation
                                        fixed = repair_offset(item["text"], span_text, start, end)
                                        if fixed is not None:
                                            ner_spans.append({**s, "start": fixed[0], "end": fixed[1]})
                                            label_stats[s["label"]] += 1
                                            label_stats["_repaired"] += 1
                                        else:
                                            label_stats["_dropped"] += 1
                        record = {
                            "id": item_id,
                            "text": item["text"],
                            "spans": svo_spans + ner_spans,
                        }
                    else:
                        record = _make_fallback(item)
                        n_fallback += 1

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_processed += 1
            else:
                n_errors += 1
                error_msg = result.get("result", {}).get("error", {}).get("message", "unknown")
                print(f"  ❌ {custom_id}: {error_msg}")
                for item in batch:
                    out.write(json.dumps(_make_fallback(item), ensure_ascii=False) + "\n")
                    n_processed += 1

    print(f"\n{'='*60}")
    print(f"✅ {n_processed} phrases traitées → {output}")
    print(f"❌ {n_errors} batches en erreur")
    print(f"⚠️  {n_fallback} phrases sans match ID (fallback)")
    if label_stats.get("_repaired"):
        print(f"🔧 {label_stats['_repaired']} spans avec offset réparé automatiquement")
    if label_stats.get("_dropped"):
        print(f"⚠️  {label_stats['_dropped']} spans irrécupérables supprimés (texte introuvable)")
    print(f"\n📊 Distribution des labels annotés :")
    for label, count in label_stats.most_common():
        if not label.startswith("_"):
            print(f"  {label:<25} {count:>6}")


def _make_fallback(item: dict) -> dict:
    """Fallback : on garde les spans tels quels (hint_* prédits + svo_*)."""
    return {
        "id": item["id"],
        "text": item["text"],
        "spans": item.get("spans", []),
        "_fallback": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL candidates (train_wiki_svo_ner.jsonl)")
    parser.add_argument("--output", required=True, help="JSONL sortie annotée par Claude")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=5, help="Phrases par requête")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--poll-interval", type=int, default=30, help="Intervalle de polling (s)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--batch-id", default=None, help="Reprendre un batch existant (skip submit)")
    parser.add_argument("--requests-file", default="data/_claude_batch_requests.jsonl")
    parser.add_argument("--results-file", default=None,
                        help="Fichier de résultats JSONL déjà téléchargé (skip submit+poll+fetch)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    # La clé n'est nécessaire que si on doit soumettre/poller/fetcher le batch
    need_api = not args.results_file
    if need_api and not api_key:
        print("❌ Clé API manquante. --api-key ou ANTHROPIC_API_KEY")
        sys.exit(1)

    # Charger les candidates
    candidates = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
            if args.max_sentences and len(candidates) >= args.max_sentences:
                break
    print(f"📝 {len(candidates)} phrases chargées")

    # Filtrer les déjà traités si output existe
    already_done = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    already_done.add(obj.get("id", ""))
                except json.JSONDecodeError:
                    continue
        if already_done:
            print(f"🔄 {len(already_done)} phrases déjà traitées, on les skip")
            candidates = [c for c in candidates if c["id"] not in already_done]
            print(f"📝 {len(candidates)} phrases restantes")

    if not candidates:
        print("✅ Rien à traiter!")
        return

    # Reconstruire les batches (toujours nécessaire pour process_results)
    batches_list = []
    for i in range(0, len(candidates), args.batch_size):
        batches_list.append(candidates[i : i + args.batch_size])

    # ── Cas : résultats déjà téléchargés localement ──
    if args.results_file:
        print(f"📂 Lecture des résultats locaux depuis {args.results_file}…")
        results = []
        with open(args.results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"📥 {len(results)} résultats chargés depuis le fichier local")
        process_results(results, batches_list, args.output)
        return

    if not args.batch_id:
        # Créer le fichier de requêtes
        create_batch_requests(candidates, args.batch_size, args.requests_file, args.model)

        # Étape 2 : Soumettre
        batch_id = submit_batch(api_key, args.requests_file)
    else:
        batch_id = args.batch_id
        print(f"🔄 Reprise du batch {batch_id}")

    # Étape 3 : Poll
    poll_batch(api_key, batch_id, args.poll_interval)

    # Étape 4 : Récupérer
    results = fetch_results(api_key, batch_id)

    # Étape 5 : Parser et écrire
    process_results(results, batches_list, args.output)


if __name__ == "__main__":
    main()

