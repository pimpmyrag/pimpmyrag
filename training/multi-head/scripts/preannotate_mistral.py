#!/usr/bin/env python3
"""
Pré-annotation / correction NER via Mistral Large — version ASYNC.
Fire 1 requête par seconde sans attendre la réponse précédente.

Usage:
  export MISTRAL_API_KEY="..."
  python3 scripts/preannotate_mistral.py \
    --input data/wikinews_ready_for_mistral.jsonl \
    --output data/wikinews_mistral_annotated.jsonl \
    --batch-size 5
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter

import httpx

# ─────────────────────────────────────────────────────────────
# TAXONOMIE (system prompt)
# ─────────────────────────────────────────────────────────────

TAXONOMY = """## Taxonomie NER — 32 labels fins groupés en 8 catégories

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
- **hint_quantity** : quantité avec unité ("15 kilomètres", "trois tonnes", "200 mégawatts")
- **hint_measure** : mesure physique ("37°C", "magnitude 6,2", "120 dB")
- **hint_percentage** : pourcentage ("45 %", "un tiers")
- **hint_count** : nombre de choses ("trois personnes", "200 soldats")
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


def build_user_prompt(batch: list[dict]) -> str:
    """Construit le prompt utilisateur avec les phrases et pré-annotations."""
    parts = []
    for item in batch:
        # Simplifier les prédictions pour le prompt
        preds = []
        for p in item.get("predictions", []):
            preds.append({
                "label": p.get("fine", ""),
                "start": p.get("char_start", 0),
                "end": p.get("char_end", 0),
                "text": p.get("text", ""),
            })

        parts.append(
            f'ID: {item["id"]}\n'
            f'Phrase: "{item["text"]}"\n'
            f'Pré-annotations: {json.dumps(preds, ensure_ascii=False)}'
        )

    return "Corrige les annotations suivantes :\n\n" + "\n\n".join(parts)


def parse_response(response_text: str, batch: list[dict]) -> list[dict]:
    """Parse la réponse JSON de Mistral."""
    # Nettoyer la réponse (enlever les ```json ... ``` si présent)
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        results = json.loads(text)
        if isinstance(results, dict):
            results = [results]
        return results
    except json.JSONDecodeError:
        # Essayer de trouver le JSON dans la réponse
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"  ⚠️  Impossible de parser la réponse JSON")
        return []


VALID_LABELS = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_infra", "hint_weapon", "hint_vehicle", "hint_substance",
    "hint_food", "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named", "hint_time_date",
    "hint_time_clock", "hint_time_duration", "hint_quantity", "hint_measure",
    "hint_percentage", "hint_count", "hint_money", "hint_rate",
    "hint_law", "hint_work_of_art", "hint_concept", "hint_disease", "hint_language",
}


async def call_mistral_async(client: httpx.AsyncClient, api_key: str, model: str,
                              system: str, user: str, max_retries: int = 8) -> str:
    """Appel Mistral async avec retry exponentiel."""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 60)
                print(f"  ⏳ 429 rate limit, attente {wait}s (tentative {attempt+1})")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if "429" in str(e):
                wait = min(2 ** (attempt + 1), 60)
                print(f"  ⏳ Rate limit, attente {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  ❌ Erreur: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    raise Exception(f"Échec après {max_retries} tentatives")


def process_response(response_text: str, batch: list[dict], label_stats: Counter) -> list[dict]:
    """Parse la réponse et produit les records à écrire."""
    results = parse_response(response_text, batch)
    result_by_id = {r["id"]: r for r in results if "id" in r}
    records = []

    for item in batch:
        item_id = item["id"]
        if item_id in result_by_id:
            corrected = result_by_id[item_id]
            valid_spans = []
            for s in corrected.get("spans", []):
                if all(k in s for k in ("label", "start", "end", "text")):
                    if s["label"] in VALID_LABELS:
                        valid_spans.append(s)
                        label_stats[s["label"]] += 1
            records.append({
                "id": item_id,
                "text": item["text"],
                "spans": valid_spans,
                "source_title": item.get("source_title", ""),
            })
        else:
            print(f"  ⚠️  Pas de résultat pour {item_id}")
            records.append(make_fallback(item))

    return records


def make_fallback(item: dict) -> dict:
    """Crée un record fallback à partir des pré-annotations."""
    preds_as_spans = [
        {"label": p.get("fine",""), "start": p.get("char_start",0),
         "end": p.get("char_end",0), "text": p.get("text","")}
        for p in item.get("predictions", [])
    ]
    return {
        "id": item["id"],
        "text": item["text"],
        "spans": preds_as_spans,
        "source_title": item.get("source_title", ""),
        "_fallback": True,
    }


class RateLimiter:
    """Garantit au moins `interval` secondes entre chaque appel."""
    def __init__(self, interval: float):
        self._interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


async def worker(queue: asyncio.Queue, results_queue: asyncio.Queue,
                 client: httpx.AsyncClient, api_key: str, model: str,
                 rate_limiter: RateLimiter):
    """Worker qui consomme les batches de la queue et envoie les requêtes."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        batch_idx, batch = item
        try:
            user_prompt = build_user_prompt(batch)
            await rate_limiter.acquire()
            response_text = await call_mistral_async(client, api_key, model, SYSTEM_PROMPT, user_prompt)
            await results_queue.put((batch_idx, batch, response_text, None))
        except Exception as e:
            await results_queue.put((batch_idx, batch, None, e))
        queue.task_done()


async def async_main(args):
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Clé API manquante. Utilise --api-key ou export MISTRAL_API_KEY=...")
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

    # Resume
    already_done = set()
    if args.resume and os.path.exists(args.output):
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
        print(f"🔄 Resume: {len(already_done)} phrases déjà traitées")
        candidates = [c for c in candidates if c["id"] not in already_done]
        print(f"📝 {len(candidates)} phrases restantes")

    if not candidates:
        print("✅ Rien à traiter!")
        return

    # Préparer les batches
    batches = []
    for i in range(0, len(candidates), args.batch_size):
        batches.append((i, candidates[i : i + args.batch_size]))
    total_batches = len(batches)
    print(f"📦 {total_batches} batches de {args.batch_size} phrases")

    mode = "a" if args.resume and already_done else "w"
    out_file = open(args.output, mode, encoding="utf-8")

    label_stats = Counter()
    n_processed = 0
    n_errors = 0
    t_start = time.time()

    # Nombre de requêtes en vol simultanément
    # Avec 1 req/s et ~15s de latence, on peut avoir ~15 en vol
    max_in_flight = args.concurrency

    queue = asyncio.Queue()
    results_queue = asyncio.Queue()
    rate_limiter = RateLimiter(args.delay)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Lancer les workers
        workers = []
        for _ in range(max_in_flight):
            w = asyncio.create_task(worker(queue, results_queue, client, api_key, args.model, rate_limiter))
            workers.append(w)

        # Feeder : enqueue tous les batches d'un coup, le rate limiter gère le débit
        async def feeder():
            for batch_item in batches:
                await queue.put(batch_item)
            # Signaler la fin aux workers
            for _ in range(max_in_flight):
                await queue.put(None)

        feeder_task = asyncio.create_task(feeder())

        # Collecter les résultats
        n_expected = len(batches)
        n_received = 0
        while n_received < n_expected:
            try:
                batch_idx, batch, response_text, error = await asyncio.wait_for(
                    results_queue.get(), timeout=180.0
                )
            except asyncio.TimeoutError:
                print("  ⚠️  Timeout en attente de résultat, on continue...")
                continue

            n_received += 1

            if error:
                print(f"  ❌ Erreur batch {batch_idx}: {error}")
                n_errors += 1
                for item in batch:
                    record = make_fallback(item)
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_processed += 1
            else:
                records = process_response(response_text, batch, label_stats)
                for record in records:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_processed += 1

            out_file.flush()

            # Progress toutes les 10 réponses
            if n_received % 10 == 0 or n_received == n_expected:
                elapsed = time.time() - t_start
                rate = n_processed / elapsed if elapsed > 0 else 0
                remaining = len(candidates) - n_processed
                eta = remaining / rate if rate > 0 else 0
                print(f"  {n_processed}/{len(candidates)} ({n_processed/len(candidates)*100:.1f}%) | "
                      f"{n_errors} err | {elapsed:.0f}s | ETA {eta:.0f}s | "
                      f"{n_received}/{n_expected} batches")

        await feeder_task
        await asyncio.gather(*workers)

    out_file.close()

    print(f"\n{'='*60}")
    print(f"✅ {n_processed} phrases traitées → {args.output}")
    print(f"❌ {n_errors} erreurs (fallback)")
    print(f"⏱️  Temps total: {time.time() - t_start:.0f}s")
    print(f"🚀 Débit: {n_processed / (time.time() - t_start):.1f} phrases/s")
    print(f"\n📊 Distribution des labels annotés par Mistral :")
    for label, count in label_stats.most_common():
        print(f"  {label:<25} {count:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0, help="Délai entre envois (secondes)")
    parser.add_argument("--concurrency", type=int, default=15, help="Requêtes en vol max")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

