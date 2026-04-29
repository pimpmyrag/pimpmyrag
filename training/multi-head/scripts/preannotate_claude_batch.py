#!/usr/bin/env python3
"""
Pré-annotation NER + SVO roles + morpho via Claude Batch API.

Annote pour chaque phrase :
  - Spans NER hint_* (corrigés)  avec gender/number/svo_role/gov_verb_start
  - Spans verb_trigger (verbes d'action gouverneurs)
  - Spans pronoms pron_subj/pron_obj avec gender/number/person
    → signaux forts pour coréférence asynchrone ultérieure

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

# ─── Taxonomie NER ───────────────────────────────────────────────────────────

TAXONOMY = """## Taxonomie NER — 31 labels fins groupés en 8 catégories

### PER (Personnes)
- **hint_person_name** : nom propre de personne ("Emmanuel Macron", "Jean Dupont")
- **hint_person_role** : rôle, titre, fonction ("président", "ministre", "médecin") — SANS le nom propre
- **hint_norp** : nationalité, ethnie, religion, parti politique ("français", "catholiques", "républicains")
- **hint_group_role** : groupe de personnes par rôle ("les soldats", "les manifestants", "la police")

### LOC (Lieux)
- **hint_gpe** : entité géopolitique — pays, ville, région ("France", "Paris", "Bretagne")
- **hint_fac_name** : bâtiment, monument, installation nommée ("Tour Eiffel", "gare de Lyon")
- **hint_loc_generic** : lieu générique non-GPE ("la frontière", "la côte", "le centre-ville")
- **hint_infra** : infrastructure ("autoroute", "pont", "voie ferrée", "pipeline")

### ORG (Organisations)
- **hint_org_name** : nom d'organisation ("ONU", "Apple", "Médecins sans frontières")

### TIME (Temps)
- **hint_time_date** : date, jour, année ("14 juillet 2024", "lundi", "2023")
- **hint_time_clock** : heure précise ("13h00", "midi", "18 h 46")
- **hint_time_duration** : durée ("trois jours", "depuis 2020", "pendant six mois")

### EVENT (Événements)
- **hint_event_nominal** : événement décrit par un nom commun ("élection", "attentat", "crise")
- **hint_event_named** : événement nommé ("Jeux olympiques de Paris 2024", "COP28")

### OBJECT (Objets)
- **hint_weapon** : arme ("fusil", "missile", "couteau", "bombe")
- **hint_vehicle** : véhicule ("avion", "voiture", "navire", "TGV")
- **hint_substance** : substance, matière première ("pétrole", "uranium", "chlore")
- **hint_food** : aliment, boisson ("blé", "vin", "fromage", "café")
- **hint_tool** : outil, instrument, appareil ("radar", "téléphone", "scanner")
- **hint_object_generic** : objet physique autre ("drapeau", "colis", "document")
- **hint_object_name** : objet nommé / marque ("iPhone", "Rafale", "Falcon 9")

### VALUE (Valeurs)
- **hint_measure** : mesure physique avec unité ("35 nœuds", "52 000 m³", "37°C")
- **hint_percentage** : pourcentage ("45 %", "un tiers")
- **hint_count** : nombre de choses ou personnes ("trois personnes", "200 soldats")
- **hint_money** : montant monétaire ("15 millions d'euros", "2,5 milliards de dollars")
- **hint_rate** : taux, ratio ("3,5 %", "1 pour 1000")

### ABSTRACT (Abstraits)
- **hint_law** : loi, traité, texte juridique ("RGPD", "loi Climat", "article 49.3")
- **hint_work_of_art** : œuvre d'art, livre, film ("Les Misérables", "La Joconde")
- **hint_concept** : concept abstrait ("démocratie", "laïcité", "souveraineté")
- **hint_disease** : maladie, pathologie ("Covid-19", "grippe aviaire", "cancer")
- **hint_language** : langue ("français", "mandarin", "arabe")

## Règles NER importantes
1. Les spans ne doivent PAS se chevaucher (mais l'imbrication est autorisée si types différents)
2. "le président Macron" → annoter SÉPARÉMENT : "président" = hint_person_role, "Macron" = hint_person_name
3. Les déterminants (le, la, les, un, une) ne font PAS partie du span sauf s'ils sont indissociables
4. Annoter TOUTES les entités de la phrase
5. "start" = index du premier caractère, "end" = index APRÈS le dernier caractère
"""

SVO_MORPHO_GUIDE = """
## Annotation SVO et morphologie

### A. Verb triggers (label: "verb_trigger")
Annote les verbes d'action qui gouvernent des participants NER.
- Inclure l'auxiliaire si présent : "a déclaré", "ont été tués", "sera nommé"
- Ne pas annoter les verbes copules purs ("est", "sont") sauf si sémantiquement forts

Champs OBLIGATOIRES sur chaque verb_trigger :
- "voice": "active" | "passive"
  → passive = conjugaison avec être + participe passé ("a été arrêté", "ont été tués", "sera nommé")
  → active = tous les autres cas
- "negated": true  (omettre si false — ne pas écrire "negated": false)
  → true si le verbe est nié : "n'a pas signé", "ne peut pas", "sans avoir déclaré"
- "certainty": "certain" | "modal" | "denied"
  → certain  : fait présenté comme réel ("a signé", "est parti")
  → modal    : possibilité/obligation ("pourrait signer", "devrait partir", "peut-être")
  → denied   : négation sémantique ("n'a pas signé", "refuse de") — combiner avec negated:true

### B. Rôle SVO sur les spans NER (champ optionnel "svo_role")
Pour chaque span hint_* qui est argument d'un trigger, ajouter :
- "svo_role": "SUBJECT" | "OBJECT" | "OBLIQUE" | "APPOS" | "NONE"
- "gov_verb_start": int  ← position start du verb_trigger OR du hint_event_nominal gouverneur

Définitions :
- SUBJECT       : sujet grammatical ("Macron a signé" → Macron=SUBJECT)
                  ⚠️ En voix PASSIVE, le SUBJECT grammatical est le PATIENT sémantique
- OBJECT        : objet direct ("a signé la loi" → loi=OBJECT)
- OBLIQUE       : complément circonstanciel de lieu/temps/manière ("à Paris", "hier", "violemment")
- OBLIQUE_AGENT : agent sémantique en construction passive, introduit par "par"
                  "a été arrêté PAR LA POLICE" → "police"=OBLIQUE_AGENT  (semantic agent !)
                  "a été signé PAR MACRON"     → "Macron"=OBLIQUE_AGENT
- OBLIQUE_CAUSE : complément causal ("à cause de la crise", "suite à l'attentat", "en raison de…")
- APPOS         : apposition explicative ("Macron, le président" → "président"=APPOS de Macron)
- NONE          : entité sans lien syntaxique direct avec un verbe

Modificateurs nominaux (champ "mod_of_start") :
Si un span est un GÉNITIF ou MODIFICATEUR D'UN AUTRE SPAN (pas argument du verbe), ajouter :
- "mod_of_start": int  ← position start du span NER dont ce span est modificateur
- NE PAS mettre svo_role sur ce span (ou mettre NONE)
Exemples :
  "pompier de Paris" → "Paris" a mod_of_start=4 (start de "pompier"), pas de svo_role
  "tribunal de Lyon" → "Lyon" a mod_of_start=9
  "ministre de l'Économie" → "Économie" a mod_of_start=9
  "forces de l'ordre" → "l'ordre" est dans le span "forces de l'ordre", pas de span séparé ici

Coordination : si plusieurs spans sont SUBJECT/OBJECT du même verbe, chacun a son propre
gov_verb_start identique. Ex: "Macron et Scholz ont signé" → les deux = SUBJECT, gov_verb_start=15.

Clauses relatives : si un span est argument d'un verbe VIA une relative, annoter normalement
avec gov_verb_start pointant vers le verbe de la relative.
Ex: "Le pompier qui a secouru la victime" →
  "pompier" a svo_role=SUBJECT, gov_verb_start=start("a secouru")
  "victime" a svo_role=OBJECT, gov_verb_start=start("a secouru")

Nominalisations : si un hint_event_nominal joue le rôle de trigger implicite (pas de verb_trigger),
utiliser gov_verb_start pointant vers le start du hint_event_nominal.
Ex: "L'arrestation de Dupont par la police" →
  "Dupont" : svo_role=OBJECT, gov_verb_start=2 (start de "arrestation")
  "police" : svo_role=OBLIQUE_AGENT, gov_verb_start=2

### C. Morphologie sur les spans NER (champs "gender", "number")
Pour PER (hint_person_name, hint_person_role, hint_norp, hint_group_role),
ORG (hint_org_name), EVENT (hint_event_nominal, hint_event_named) :
- "gender": "M" | "F" | "N"  (N = neutre/indéterminé)
- "number": "SG" | "PL"

Signaux pour CORÉFÉRENCE ULTÉRIEURE. Ne pas annoter sur TIME/VALUE/LOC sauf cas évident.

### D. Pronoms (labels "pron_subj", "pron_obj")
- "pron_subj" : pronom sujet ("il", "elle", "ils", "elles", "on", "ce", "cela", "celui-ci")
- "pron_obj"  : pronom objet clitique ("le", "la", "les", "lui", "leur", "y", "en")

Champs OBLIGATOIRES sur chaque pronom :
- "gender": "M" | "F" | "N"
- "number": "SG" | "PL"
- "person": "1" | "2" | "3"
- "svo_role": "SUBJECT" | "OBJECT" | "OBLIQUE"
- "gov_verb_start": int

NE PAS résoudre l'antécédent — uniquement les traits morpho + lien syntaxique.

### Exemples complets

Phrase: "Macron a déclaré la guerre en Ukraine."
Spans:
[
  {"label": "verb_trigger", "start": 7, "end": 18, "text": "a déclaré", "voice": "active", "certainty": "certain"},
  {"label": "hint_person_name", "start": 0, "end": 6, "text": "Macron",
   "gender": "M", "number": "SG", "svo_role": "SUBJECT", "gov_verb_start": 7},
  {"label": "hint_event_nominal", "start": 19, "end": 28, "text": "la guerre",
   "gender": "F", "number": "SG", "svo_role": "OBJECT", "gov_verb_start": 7},
  {"label": "hint_gpe", "start": 32, "end": 39, "text": "Ukraine",
   "svo_role": "OBLIQUE", "gov_verb_start": 7}
]

Phrase: "Dupont a été arrêté par la police. Il n'a pas résisté."
Spans:
[
  {"label": "hint_person_name", "start": 0, "end": 6, "text": "Dupont",
   "gender": "M", "number": "SG", "svo_role": "SUBJECT", "gov_verb_start": 7},
  {"label": "verb_trigger", "start": 7, "end": 22, "text": "a été arrêté", "voice": "passive", "certainty": "certain"},
  {"label": "hint_group_role", "start": 27, "end": 34, "text": "police",
   "gender": "F", "number": "SG", "svo_role": "OBLIQUE_AGENT", "gov_verb_start": 7},
  {"label": "pron_subj", "start": 36, "end": 38, "text": "Il",
   "gender": "M", "number": "SG", "person": "3", "svo_role": "SUBJECT", "gov_verb_start": 39},
  {"label": "verb_trigger", "start": 39, "end": 51, "text": "n'a pas résisté", "voice": "active", "certainty": "denied", "negated": true}
]

Phrase: "Le pompier de Paris a effectué un sauvetage à Marseille hier."
Spans:
[
  {"label": "hint_group_role", "start": 3, "end": 10, "text": "pompier",
   "gender": "M", "number": "SG", "svo_role": "SUBJECT", "gov_verb_start": 19},
  {"label": "hint_gpe", "start": 14, "end": 19, "text": "Paris",
   "mod_of_start": 3},
  {"label": "verb_trigger", "start": 19, "end": 28, "text": "a effectué", "voice": "active", "certainty": "certain"},
  {"label": "hint_event_nominal", "start": 32, "end": 41, "text": "sauvetage",
   "gender": "M", "number": "SG", "svo_role": "OBJECT", "gov_verb_start": 19},
  {"label": "hint_gpe", "start": 44, "end": 52, "text": "Marseille",
   "svo_role": "OBLIQUE", "gov_verb_start": 19},
  {"label": "hint_time_date", "start": 53, "end": 57, "text": "hier",
   "svo_role": "OBLIQUE", "gov_verb_start": 19}
]

Phrase: "Les soldats ont été tués. Ils étaient stationnés à Kaboul."
Spans:
[
  {"label": "hint_group_role", "start": 4, "end": 11, "text": "soldats",
   "gender": "M", "number": "PL", "svo_role": "SUBJECT", "gov_verb_start": 12},
  {"label": "verb_trigger", "start": 12, "end": 24, "text": "ont été tués", "voice": "passive", "certainty": "certain"},
  {"label": "pron_subj", "start": 26, "end": 29, "text": "Ils",
   "gender": "M", "number": "PL", "person": "3", "svo_role": "SUBJECT", "gov_verb_start": 30},
  {"label": "verb_trigger", "start": 30, "end": 43, "text": "étaient stationnés", "voice": "passive", "certainty": "certain"},
  {"label": "hint_gpe", "start": 46, "end": 52, "text": "Kaboul",
   "svo_role": "OBLIQUE", "gov_verb_start": 30}
]

Phrase: "L'arrestation de Dupont par les forces de l'ordre a provoqué des manifestations."
Spans:
[
  {"label": "hint_event_nominal", "start": 2, "end": 13, "text": "arrestation",
   "gender": "F", "number": "SG"},
  {"label": "hint_person_name", "start": 17, "end": 23, "text": "Dupont",
   "gender": "M", "number": "SG", "svo_role": "OBJECT", "gov_verb_start": 2},
  {"label": "hint_group_role", "start": 28, "end": 46, "text": "forces de l'ordre",
   "gender": "F", "number": "PL", "svo_role": "OBLIQUE", "gov_verb_start": 2},
  {"label": "verb_trigger", "start": 47, "end": 57, "text": "a provoqué", "voice": "active", "certainty": "certain"},
  {"label": "hint_event_nominal", "start": 62, "end": 76, "text": "manifestations",
   "gender": "F", "number": "PL", "svo_role": "OBJECT", "gov_verb_start": 47}
]
"""

EXAMPLES = """## Exemple avec coordination et modalité

Phrase: "Macron et Scholz pourraient signer un accord à Berlin."
Spans: [
  {"label": "hint_person_name", "start": 0, "end": 6, "text": "Macron",
   "gender": "M", "number": "SG", "svo_role": "SUBJECT", "gov_verb_start": 18},
  {"label": "hint_person_name", "start": 10, "end": 16, "text": "Scholz",
   "gender": "M", "number": "SG", "svo_role": "SUBJECT", "gov_verb_start": 18},
  {"label": "verb_trigger", "start": 17, "end": 27, "text": "pourraient signer", "voice": "active", "certainty": "modal"},
  {"label": "hint_event_nominal", "start": 33, "end": 39, "text": "accord",
   "gender": "M", "number": "SG", "svo_role": "OBJECT", "gov_verb_start": 17},
  {"label": "hint_gpe", "start": 42, "end": 48, "text": "Berlin",
   "svo_role": "OBLIQUE", "gov_verb_start": 17}
]
"""

SYSTEM_PROMPT = f"""Tu es un expert en annotation NER, syntaxe et morphologie pour le français.

{TAXONOMY}

{SVO_MORPHO_GUIDE}

{EXAMPLES}

## Ta tâche
Pour chaque phrase, tu reçois des pré-annotations NER automatiques.
Tu dois :
1. CORRIGER les labels NER erronés, frontières trop larges/étroites, faux positifs
2. AJOUTER les entités manquées par le modèle
3. AJOUTER les verb_trigger avec voice + certainty (+ negated si nié)
4. AJOUTER svo_role + gov_verb_start sur chaque span NER argumentel
   (gov_verb_start peut pointer vers un verb_trigger OU un hint_event_nominal)
5. AJOUTER gender + number sur les spans PER/ORG/EVENT
6. AJOUTER les pronoms pron_subj/pron_obj avec gender/number/person/svo_role/gov_verb_start

Retourne UNIQUEMENT un tableau JSON valide.
Chaque objet : {{"id": "...", "text": "...", "spans": [...]}}
Chaque span a au minimum : {{"label": "...", "start": int, "end": int, "text": "..."}}
verb_trigger a TOUJOURS : voice + certainty (+ negated:true si applicable).
Les autres champs (gender, number, person, svo_role, gov_verb_start) uniquement si pertinents.

IMPORTANT : text[start:end] doit correspondre EXACTEMENT à span["text"].
Ne retourne RIEN d'autre que le JSON.
"""


# ─── Labels valides ───────────────────────────────────────────────────────────

VALID_NER_LABELS = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_infra", "hint_weapon", "hint_vehicle", "hint_substance",
    "hint_food", "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named", "hint_time_date",
    "hint_time_clock", "hint_time_duration", "hint_measure",
    "hint_percentage", "hint_count", "hint_money", "hint_rate",
    "hint_law", "hint_work_of_art", "hint_concept", "hint_disease", "hint_language",
}
VALID_SVO_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}
VALID_LABELS = VALID_NER_LABELS | VALID_SVO_LABELS

VALID_ROLES = {"SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE", "APPOS", "NONE"}
VALID_GENDER = {"M", "F", "N"}
VALID_NUMBER = {"SG", "PL"}
VALID_PERSON = {"1", "2", "3"}
VALID_VOICE = {"active", "passive"}
VALID_CERTAINTY = {"certain", "modal", "denied"}


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


def validate_span_extras(s: dict) -> dict:
    """Filtre et valide les champs optionnels d'un span."""
    clean = {k: s[k] for k in ("label", "start", "end", "text")}
    # SVO
    if s.get("svo_role") in VALID_ROLES:
        clean["svo_role"] = s["svo_role"]
    if isinstance(s.get("gov_verb_start"), int):
        clean["gov_verb_start"] = s["gov_verb_start"]
    if isinstance(s.get("mod_of_start"), int):
        clean["mod_of_start"] = s["mod_of_start"]
    # Morpho
    if s.get("gender") in VALID_GENDER:
        clean["gender"] = s["gender"]
    if s.get("number") in VALID_NUMBER:
        clean["number"] = s["number"]
    if s.get("person") in VALID_PERSON:
        clean["person"] = s["person"]
    # Verb trigger fields
    if s.get("voice") in VALID_VOICE:
        clean["voice"] = s["voice"]
    if s.get("certainty") in VALID_CERTAINTY:
        clean["certainty"] = s["certainty"]
    if s.get("negated") is True:
        clean["negated"] = True
    return clean


def build_user_prompt(batch: list[dict]) -> str:
    parts = []
    for item in batch:
        preds = [
            {"label": sp["label"], "start": sp["start"], "end": sp["end"], "text": sp["text"]}
            for sp in item.get("spans", [])
            if sp.get("label", "").startswith("hint_")
        ]
        parts.append(
            f'ID: {item["id"]}\n'
            f'Phrase: "{item["text"]}"\n'
            f'Pré-annotations NER: {json.dumps(preds, ensure_ascii=False)}'
        )
    return "Corrige et enrichis les annotations suivantes :\n\n" + "\n\n".join(parts)


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


# ─── Étape 1 : Créer le fichier JSONL de requêtes ────────────────────────────

def create_batch_requests(candidates: list[dict], batch_size: int, output_jsonl: str,
                          args_model: str = "claude-sonnet-4-6"):
    batches = []
    for i in range(0, len(candidates), batch_size):
        batches.append(candidates[i: i + batch_size])

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batches):
            user_prompt = build_user_prompt(batch)
            request = {
                "custom_id": f"batch_{batch_idx}",
                "params": {
                    "model": args_model,
                    "max_tokens": 8192,
                    "temperature": 0.1,
                    "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                    "messages": [{"role": "user", "content": user_prompt}],
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"📦 {len(batches)} requêtes batch → {output_jsonl}")
    return batches


# ─── Étape 2 : Soumettre le batch ────────────────────────────────────────────

def _batch_id_file(requests_jsonl: str) -> str:
    """Chemin du fichier .batch_id associé à un fichier de requêtes."""
    return requests_jsonl.replace(".jsonl", ".batch_id")

def _save_batch_id(batch_id: str, requests_jsonl: str):
    path = _batch_id_file(requests_jsonl)
    with open(path, "w") as f:
        f.write(batch_id)
    print(f"💾 Batch ID sauvegardé dans {path}")

def _load_batch_id(requests_jsonl: str) -> str | None:
    path = _batch_id_file(requests_jsonl)
    if os.path.exists(path):
        with open(path) as f:
            bid = f.read().strip()
        if bid:
            return bid
    return None

def submit_batch(api_key: str, requests_jsonl: str, max_retries: int = 6) -> str:
    url = "https://api.anthropic.com/v1/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "message-batches-2024-09-24",
    }
    requests_list = []
    with open(requests_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests_list.append(json.loads(line))

    print(f"📤 Envoi de {len(requests_list)} requêtes au Batch API...")

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json={"requests": requests_list})
                resp.raise_for_status()
                data = resp.json()

            batch_id = data["id"]
            # Sauvegarde locale de l'ID pour pouvoir reprendre en cas de crash
            _save_batch_id(batch_id, requests_jsonl)
            print(f"✅ Batch créé : {batch_id}  |  Status: {data.get('processing_status', 'unknown')}")
            return batch_id

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            # Retry uniquement sur erreurs serveur transitoires (5xx)
            if status_code >= 500:
                wait = min(10 * 2 ** (attempt - 1), 120)   # 10s, 20s, 40s, 80s, 120s…
                print(f"⚠️  Tentative {attempt}/{max_retries} — HTTP {status_code}, retry dans {wait}s…")
                last_exc = e
                time.sleep(wait)
            else:
                raise   # 4xx → pas de retry (auth, quota…)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = min(10 * 2 ** (attempt - 1), 120)
            print(f"⚠️  Tentative {attempt}/{max_retries} — réseau ({type(e).__name__}), retry dans {wait}s…")
            last_exc = e
            time.sleep(wait)

    raise RuntimeError(
        f"❌ submit_batch échoué après {max_retries} tentatives. Dernière erreur : {last_exc}\n"
        f"💡 Astuce : relance avec --requests-file {requests_jsonl} pour éviter de recréer le fichier, "
        f"ou --batch-id <id> si le batch a quand même été créé côté Anthropic."
    )


# ─── Étape 3 : Poll ──────────────────────────────────────────────────────────

def poll_batch(api_key: str, batch_id: str, poll_interval: int = 30) -> dict:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
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
            print(f"  ⚠️  Erreur poll ({type(e).__name__}), retry dans {wait}s… ({consecutive_errors} consécutives)")
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
        "anthropic-beta": "message-batches-2024-09-24",
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
            print(f"⚠️  fetch_results tentative {attempt}/{max_retries} — {type(e).__name__}, retry dans {wait}s…")
            last_exc = e
            time.sleep(wait)
    raise RuntimeError(f"❌ fetch_results échoué après {max_retries} tentatives : {last_exc}")


# ─── Étape 5 : Parser et écrire ──────────────────────────────────────────────

def process_results(results: list[dict], batches: list[list[dict]], output: str):
    """
    Parse les résultats batch et écrit le JSONL final.
    Spans retenus :
      - hint_*  : corrigés par Claude + gender/number/svo_role/gov_verb_start si présents
      - verb_trigger : nouveaux, annotés par Claude
      - pron_subj/pron_obj : nouveaux, annotés par Claude avec morpho
    Les anciens spans svo_*/pron_* du silver Stanza sont SUPPRIMÉS (remplacés par les annotations Claude).
    """
    batch_by_id = {f"batch_{i}": batch for i, batch in enumerate(batches)}

    label_stats = Counter()
    role_stats = Counter()
    morpho_stats = Counter()
    voice_stats = Counter()
    certainty_stats = Counter()
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
                response_text = "".join(
                    b["text"] for b in message.get("content", []) if b.get("type") == "text"
                )
                parsed = parse_response(response_text)
                result_by_id = {r["id"]: r for r in parsed if "id" in r}

                for item in batch:
                    item_id = item["id"]

                    if item_id in result_by_id:
                        corrected = result_by_id[item_id]
                        spans_out = []
                        for s in corrected.get("spans", []):
                            if not all(k in s for k in ("label", "start", "end", "text")):
                                continue
                            if s["label"] not in VALID_LABELS:
                                label_stats["_unknown_label"] += 1
                                continue

                            span_text = s["text"]
                            start, end = s["start"], s["end"]

                            # Vérifier/réparer l'offset
                            if item["text"][start:end] != span_text:
                                fixed = repair_offset(item["text"], span_text, start, end)
                                if fixed is not None:
                                    start, end = fixed
                                    label_stats["_repaired"] += 1
                                else:
                                    label_stats["_dropped"] += 1
                                    continue

                            clean = validate_span_extras({**s, "start": start, "end": end})
                            spans_out.append(clean)
                            label_stats[clean["label"]] += 1
                            if "svo_role" in clean:
                                role_stats[clean["svo_role"]] += 1
                            if "gender" in clean:
                                morpho_stats[f"gender_{clean['gender']}"] += 1
                            if "number" in clean:
                                morpho_stats[f"number_{clean['number']}"] += 1
                            if "voice" in clean:
                                voice_stats[clean["voice"]] += 1
                            if "certainty" in clean:
                                certainty_stats[clean["certainty"]] += 1
                            if clean.get("negated"):
                                certainty_stats["negated"] += 1

                        record = {"id": item_id, "text": item["text"], "spans": spans_out}
                    else:
                        # Fallback : garder les hints du silver, sans les svo_* bruités
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

    print(f"\n{'=' * 60}")
    print(f"✅ {n_processed} phrases → {output}")
    print(f"❌ {n_errors} batches en erreur  |  ⚠️  {n_fallback} fallbacks")
    if label_stats.get("_repaired"):
        print(f"🔧 {label_stats['_repaired']} offsets réparés")
    if label_stats.get("_dropped"):
        print(f"⚠️  {label_stats['_dropped']} spans irrécupérables supprimés")

    print(f"\n📊 Labels NER :")
    for label, count in label_stats.most_common():
        if not label.startswith("_") and label in VALID_NER_LABELS:
            print(f"  {label:<28} {count:>6}")

    print(f"\n🔗 Verb triggers : {label_stats.get('verb_trigger', 0)}")
    print(f"👤 Pronoms : subj={label_stats.get('pron_subj', 0)}  obj={label_stats.get('pron_obj', 0)}")

    print(f"\n🎭 Rôles SVO :")
    for role, count in role_stats.most_common():
        print(f"  {role:<12} {count:>6}")

    print(f"\n🔊 Voix (verb_trigger) :")
    for v, count in voice_stats.most_common():
        print(f"  {v:<10} {count:>6}")

    print(f"\n🔮 Modalité (verb_trigger) :")
    for c, count in certainty_stats.most_common():
        print(f"  {c:<10} {count:>6}")

    print(f"\n🔤 Morphologie :")
    for k, count in sorted(morpho_stats.items()):
        print(f"  {k:<18} {count:>6}")


def _make_fallback(item: dict) -> dict:
    """Fallback : garder uniquement les spans hint_* (pas les svo_* silver bruités)."""
    return {
        "id": item["id"],
        "text": item["text"],
        "spans": [sp for sp in item.get("spans", []) if sp.get("label", "").startswith("hint_")],
        "_fallback": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=5, help="Phrases par requête batch")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--batch-id", default=None, help="Reprendre un batch existant")
    parser.add_argument("--requests-file", default="data/_claude_batch_requests.jsonl")
    parser.add_argument("--results-file", default=None,
                        help="Résultats JSONL déjà téléchargés (skip submit+poll+fetch)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not args.results_file and not api_key:
        print("❌ Clé API manquante. --api-key ou ANTHROPIC_API_KEY")
        sys.exit(1)

    candidates = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
            if args.max_sentences and len(candidates) >= args.max_sentences:
                break
    print(f"📝 {len(candidates)} phrases chargées")

    already_done = set()
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
            candidates = [c for c in candidates if c["id"] not in already_done]
            print(f"📝 {len(candidates)} restantes")

    if not candidates:
        print("✅ Rien à traiter!")
        return

    batches_list = [candidates[i: i + args.batch_size] for i in range(0, len(candidates), args.batch_size)]

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
        process_results(results, batches_list, args.output)
        return

    if not args.batch_id:
        # Vérifie s'il existe un batch ID sauvegardé pour ce fichier de requêtes
        saved_id = _load_batch_id(args.requests_file)
        if saved_id:
            print(f"🔄 Batch ID trouvé dans le fichier de sauvegarde : {saved_id}")
            batch_id = saved_id
        else:
            create_batch_requests(candidates, args.batch_size, args.requests_file, args.model)
            batch_id = submit_batch(api_key, args.requests_file)
    else:
        batch_id = args.batch_id
        print(f"🔄 Reprise batch {batch_id}")

    poll_batch(api_key, batch_id, args.poll_interval)
    results = fetch_results(api_key, batch_id)
    process_results(results, batches_list, args.output)


if __name__ == "__main__":
    main()

