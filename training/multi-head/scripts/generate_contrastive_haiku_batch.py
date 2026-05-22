#!/usr/bin/env python3
"""
generate_contrastive_haiku_batch.py
====================================
Génère des phrases de presse française contrastives pour améliorer les labels
fins ambigus, via Claude Haiku Batch API.

Chaque paire de confusion (ex: hint_state vs hint_notion) produit des phrases
où le span cible est CLAIREMENT du label voulu, avec un contexte qui aide à
distinguer des labels voisins.

Sortie : JSONL avec phrases + NER spans (label/start/end/text uniquement).
Les annotations SVO/morpho/verb_trigger sont ajoutées ensuite par Stanza.

Usage :
    python3 scripts/generate_contrastive_haiku_batch.py \\
        --output data/contrastive_v1_raw.jsonl \\
        --n-per-pair 30 \\
        --batch-size 5 \\
        --model claude-haiku-4-5 \\
        --dry-run          # affiche les prompts sans appeler l'API

    # Reprendre un batch existant :
    python3 scripts/generate_contrastive_haiku_batch.py \\
        --output data/contrastive_v1_raw.jsonl \\
        --batch-id msgbatch_XXXX

Pipeline complet :
    1. python3 scripts/generate_contrastive_haiku_batch.py --output data/contrastive_v1_raw.jsonl
    2. python3 stanza_inject_svo_oblique_appos.py data/contrastive_v1_raw.jsonl -o data/contrastive_v1_svo.jsonl
    3. python3 scripts/review_stanza_spans_haiku_batch.py --input data/contrastive_v1_svo.jsonl --output data/contrastive_v1.jsonl
    4. Merge dans v8.19
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import re
from pathlib import Path

import httpx

# ─── Taxonomie condensée pour le prompt ───────────────────────────────────────

TAXONOMY_BRIEF = """
### Labels NER — Définitions clés

**hint_state** : État/condition DURABLE et observable (pauvreté, chômage, guerre, crise, instabilité).
  Dure dans le temps, peut s'aggraver ou s'améliorer. PAS un événement ponctuel, PAS un concept.
  Exemples : "la guerre civile", "la pauvreté", "l'instabilité politique", "la crise économique"

**hint_notion** : Concept ABSTRAIT pur, valeur, principe, idée (liberté, démocratie, souveraineté, légitimité, équité).
  Une idée qu'on peut discuter/défendre, pas une situation observable. PAS un état, PAS un domaine.
  Exemples : "la liberté d'expression", "l'État de droit", "la solidarité", "la transparence"

**hint_field** : Domaine / secteur d'activité (santé, éducation, agriculture, finance, défense, énergie).
  Désigne un SECTEUR où des activités se déroulent. PAS une notion abstraite, PAS un état.
  Exemples : "le secteur de la santé", "l'éducation nationale", "le domaine agricole"

**hint_doctrine** : Doctrine, idéologie, courant de pensée, théorie (libéralisme, islamisme, keynésianisme, féminisme).
  Un SYSTÈME DE PENSÉE nommé ou reconnaissable. PAS un concept isolé, PAS un état.
  Exemples : "le néolibéralisme", "l'islamisme radical", "le gaullisme", "la social-démocratie"

**hint_event_nominal** : Événement nommé par un NOM COMMUN, discret, limité dans le temps (élection, attentat, accident, réunion, arrêt de travail).
  A un début et une fin. PAS un état durable, PAS un concept.
  Exemples : "l'élection présidentielle", "l'attentat", "la grève", "le procès", "la manifestation"

**hint_inst_role** : Institution PUBLIQUE désignée de façon GÉNÉRIQUE, sans nom propre ("le gouvernement", "la police", "l'armée", "le parlement", "les autorités").
  On peut dire "un/une [inst_role]". PAS un nom propre/sigle → hint_inst_name.

**hint_group_role** : Groupe de personnes par RÔLE FONCTIONNEL ou situationnel ("soldats", "manifestants", "grévistes", "victimes", "militants", "civils").
  Le groupe peut changer de membres. PAS une identité ethno-culturelle → hint_norp.

**hint_work_generic** : Production culturelle/médiatique GÉNÉRIQUE sans titre (film, livre, presse, émission, série).
  Désigne un type de production. PAS une œuvre nommée → hint_work_of_art.

**hint_document** : Document administratif/officiel GÉNÉRIQUE (rapport, lettre, communiqué, contrat, données, fichier).
  Un document concret. PAS une loi → hint_law. PAS une œuvre culturelle → hint_work_generic.

**hint_object_name** : Objet nommé, marque ou modèle spécifique (iPhone, Rafale, Falcon 9, A10, Kalachnikov).
  A un NOM PROPRE. PAS générique → hint_object_generic.

**hint_object_generic** : Objet physique GÉNÉRIQUE sans nom propre (drapeau, colis, matériel, équipement, badge).
  Désigne une catégorie d'objets. PAS nommé → hint_object_name.
"""

# ─── Paires de confusion et leurs prompts ─────────────────────────────────────

CONFUSION_PAIRS = [
    {
        "id": "state_vs_notion",
        "target_label": "hint_state",
        "contrast_labels": ["hint_notion", "hint_event_nominal"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "un **hint_state** (état/condition DURABLE et observable) plutôt qu'un "
            "**hint_notion** (concept abstrait) ou un **hint_event_nominal** (événement ponctuel).\n"
            "Le span doit décrire une situation qui DURE dans le temps (jours/mois/années) "
            "et qu'on peut observer concrètement. "
            "PAS une idée abstraite comme 'liberté' ou 'démocratie'. "
            "PAS un événement qui a une date précise comme 'l'élection'."
        ),
        "examples": [
            ("La guerre civile continue de ravager le pays.", "la guerre civile", "hint_state"),
            ("Le chômage de longue durée touche désormais 15% de la population active.", "Le chômage de longue durée", "hint_state"),
            ("Face à l'instabilité politique persistante, les investisseurs fuient la région.", "l'instabilité politique persistante", "hint_state"),
        ],
    },
    {
        "id": "notion_vs_state_field",
        "target_label": "hint_notion",
        "contrast_labels": ["hint_state", "hint_field", "hint_doctrine"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "une **hint_notion** (concept abstrait pur, valeur, principe) plutôt qu'un "
            "**hint_state** (état observable), **hint_field** (secteur d'activité) ou "
            "**hint_doctrine** (idéologie nommée).\n"
            "Le span doit désigner une IDÉE qu'on peut invoquer, défendre ou remettre en question, "
            "pas une situation concrète ni un domaine d'activité."
        ),
        "examples": [
            ("Le débat sur la souveraineté nationale a animé la séance.", "la souveraineté nationale", "hint_notion"),
            ("Les militants ont brandi la transparence comme valeur fondamentale.", "la transparence", "hint_notion"),
            ("L'État de droit est au cœur des négociations avec l'Union européenne.", "L'État de droit", "hint_notion"),
        ],
    },
    {
        "id": "field_vs_notion",
        "target_label": "hint_field",
        "contrast_labels": ["hint_notion", "hint_doctrine"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "un **hint_field** (secteur/domaine d'activité concret) plutôt qu'une "
            "**hint_notion** (concept abstrait) ou une **hint_doctrine** (idéologie).\n"
            "Le span doit désigner un SECTEUR où des professionnels travaillent et des "
            "politiques publiques s'appliquent. On peut dire 'le secteur de [X]' ou 'le domaine de [X]'."
        ),
        "examples": [
            ("Le gouvernement a annoncé un plan d'investissement dans la santé publique.", "la santé publique", "hint_field"),
            ("Les réformes dans l'éducation nationale suscitent des débats.", "l'éducation nationale", "hint_field"),
            ("Les experts du domaine agricole s'alarment de la sécheresse.", "domaine agricole", "hint_field"),
        ],
    },
    {
        "id": "inst_role_vs_group_role",
        "target_label": "hint_inst_role",
        "contrast_labels": ["hint_group_role", "hint_inst_name"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "une **hint_inst_role** (institution PUBLIQUE désignée de façon générique) "
            "plutôt qu'un **hint_group_role** (groupe de personnes par rôle fonctionnel) "
            "ou une **hint_inst_name** (institution avec nom propre).\n"
            "Le span doit désigner une institution d'État/publique SANS nom propre. "
            "On peut dire 'un/une [inst_role]'. PAS un groupe de personnes."
        ),
        "examples": [
            ("La police a interpellé plusieurs suspects dans cette affaire.", "La police", "hint_inst_role"),
            ("Le parlement a adopté le texte après trois jours de débats.", "Le parlement", "hint_inst_role"),
            ("L'armée a été déployée aux frontières nord du pays.", "L'armée", "hint_inst_role"),
        ],
    },
    {
        "id": "work_generic_vs_document",
        "target_label": "hint_work_generic",
        "contrast_labels": ["hint_document", "hint_work_of_art"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "une **hint_work_generic** (production culturelle/médiatique générique SANS titre) "
            "plutôt qu'un **hint_document** (document administratif/officiel) ou "
            "une **hint_work_of_art** (œuvre nommée avec titre).\n"
            "Le span doit désigner un TYPE de production culturelle générique "
            "(un film, la presse, des séries, un livre) SANS nom propre spécifique."
        ),
        "examples": [
            ("La presse locale a largement couvert l'événement.", "La presse locale", "hint_work_generic"),
            ("Un film documentaire sur la crise a été diffusé hier soir.", "Un film documentaire", "hint_work_generic"),
            ("Les séries télévisées influencent de plus en plus l'opinion publique.", "Les séries télévisées", "hint_work_generic"),
        ],
    },
    {
        "id": "object_name_vs_generic",
        "target_label": "hint_object_name",
        "contrast_labels": ["hint_object_generic", "hint_weapon", "hint_vehicle"],
        "instruction": (
            "Génère des phrases de presse française où le span annoté est clairement "
            "un **hint_object_name** (objet avec NOM PROPRE, marque ou modèle spécifique) "
            "plutôt qu'un **hint_object_generic** (objet générique sans nom propre).\n"
            "Le span doit être un NOM PROPRE d'objet : marque commerciale, modèle militaire, "
            "désignation officielle. PAS 'un avion' mais 'un Rafale'."
        ),
        "examples": [
            ("L'armée a commandé vingt Rafale supplémentaires.", "Rafale", "hint_object_name"),
            ("Les agents étaient équipés de Tasers lors de l'intervention.", "Tasers", "hint_object_name"),
            ("SpaceX a lancé un Falcon 9 depuis Cap Canaveral.", "Falcon 9", "hint_object_name"),
        ],
    },
]

# ─── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(pair: dict, n: int) -> str:
    ex_str = "\n".join(
        f'  - "{sp}" → {lbl} dans "{text}"'
        for text, sp, lbl in pair["examples"]
    )
    return f"""{TAXONOMY_BRIEF}

---

## Tâche

{pair["instruction"]}

### Exemples de référence
{ex_str}

### Format de sortie (STRICT — JSON uniquement, pas de texte autour)

Retourne un JSON array de {n} objets, chacun avec :
- `text` : la phrase complète (20-40 mots, style presse française)
- `spans` : liste de spans annotés, chacun avec :
  - `label` : le label NER exact (parmi : {pair["target_label"]}, et éventuellement d'autres labels NER présents)
  - `text` : le texte exact du span tel qu'il apparaît dans `text`
  - `start` : offset caractère de début (inclusif, 0-indexé)
  - `end` : offset caractère de fin (exclusif)

### Règles strictes
1. Chaque phrase DOIT contenir au moins un span `{pair["target_label"]}`
2. Les offsets `start`/`end` doivent être EXACTS : `sentence["text"][start:end] == span["text"]`
3. Tu peux annoter d'autres entités présentes dans la phrase (persons, orgs, dates, etc.)
4. PAS de markdown, PAS d'explications — uniquement le JSON array
5. Style journalistique, registre neutre, français standard

### Commence maintenant :"""


# ─── API helpers ──────────────────────────────────────────────────────────────

def build_batch_requests(pairs: list[dict], n_per_pair: int, model: str) -> list[dict]:
    requests = []
    for pair in pairs:
        # On split en plusieurs requêtes si n_per_pair > 10 (limite de qualité Claude)
        chunk_size = min(n_per_pair, 8)
        n_chunks = (n_per_pair + chunk_size - 1) // chunk_size
        for chunk_idx in range(n_chunks):
            n_this = min(chunk_size, n_per_pair - chunk_idx * chunk_size)
            req_id = f"{pair['id']}_chunk{chunk_idx}"
            requests.append({
                "custom_id": req_id,
                "params": {
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "user", "content": build_prompt(pair, n_this)}
                    ],
                },
            })
    return requests


def submit_batch(requests: list[dict], api_key: str) -> str:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
            "content-type": "application/json",
        },
        json={"requests": requests},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    batch_id = data["id"]
    print(f"✅ Batch soumis : {batch_id}  ({len(requests)} requêtes)")
    return batch_id


def poll_batch(batch_id: str, api_key: str, interval: int) -> str:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }
    while True:
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data["processing_status"]
        counts = data.get("request_counts", {})
        print(f"  [{status}] {counts}", flush=True)
        if status == "ended":
            return data["results_url"]
        time.sleep(interval)


def fetch_results(results_url: str, api_key: str) -> list[dict]:
    resp = httpx.get(
        results_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = []
    for line in resp.text.strip().split("\n"):
        if line:
            results.append(json.loads(line))
    return results


# ─── Parsing et validation ─────────────────────────────────────────────────────

def parse_claude_output(custom_id: str, content: str) -> list[dict]:
    """Parse la réponse Claude (JSON array de phrases) et valide les offsets."""
    # Extraire le JSON (Claude peut wrapper dans ```json ... ```)
    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        print(f"  ⚠️  {custom_id}: aucun JSON array trouvé", file=sys.stderr)
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"  ⚠️  {custom_id}: JSON invalide — {e}", file=sys.stderr)
        return []

    valid = []
    pair_id = custom_id.rsplit("_chunk", 1)[0]

    for i, item in enumerate(items):
        text = item.get("text", "")
        spans = item.get("spans", [])
        if not text or not spans:
            continue

        fixed_spans = []
        ok = True
        for sp in spans:
            label = sp.get("label", "")
            sp_text = sp.get("text", "")
            start = sp.get("start")
            end = sp.get("end")

            if start is None or end is None:
                ok = False
                break

            # Vérification offset
            actual = text[start:end]
            if actual != sp_text:
                # Tentative de correction (find)
                idx = text.find(sp_text)
                if idx >= 0:
                    start, end = idx, idx + len(sp_text)
                else:
                    print(f"  ⚠️  {custom_id}[{i}] offset mismatch: '{sp_text}' vs '{actual}'", file=sys.stderr)
                    ok = False
                    break

            fixed_spans.append({
                "label": label,
                "text": sp_text,
                "start": start,
                "end": end,
                "_label_src": "generated_contrastive",
                "_pair_id": pair_id,
            })

        if ok and fixed_spans:
            valid.append({
                "id": f"contrastive_{pair_id}_{i:03d}",
                "text": text,
                "spans": fixed_spans,
                "_source": "contrastive_generated",
                "_pair_id": pair_id,
            })

    return valid


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Génération phrases contrastives via Claude Haiku Batch API")
    ap.add_argument("--output", required=True, help="Fichier JSONL de sortie")
    ap.add_argument("--n-per-pair", type=int, default=30, help="Nombre de phrases par paire de confusion (défaut: 30)")
    ap.add_argument("--batch-size", type=int, default=5, help="Non utilisé (kept for compat)")
    ap.add_argument("--model", default="claude-haiku-4-5", help="Modèle Claude (défaut: claude-haiku-4-5)")
    ap.add_argument("--pairs", default=None, help="IDs de paires séparés par virgule (défaut: toutes). Ex: state_vs_notion,field_vs_notion")
    ap.add_argument("--poll-interval", type=int, default=30, help="Secondes entre deux polls (défaut: 30)")
    ap.add_argument("--batch-id", default=None, help="Reprendre un batch existant")
    ap.add_argument("--requests-file", default=None, help="Sauvegarder les requêtes dans ce fichier JSONL")
    ap.add_argument("--dry-run", action="store_true", help="Affiche les prompts sans appeler l'API")
    ap.add_argument("--api-key", default=None, help="Clé API Anthropic (défaut: ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("❌ ANTHROPIC_API_KEY manquant", file=sys.stderr)
        sys.exit(1)

    # Filtrer les paires si demandé
    pairs = CONFUSION_PAIRS
    if args.pairs:
        wanted = set(args.pairs.split(","))
        pairs = [p for p in CONFUSION_PAIRS if p["id"] in wanted]
        if not pairs:
            print(f"❌ Aucune paire trouvée parmi : {args.pairs}", file=sys.stderr)
            sys.exit(1)

    print(f"🎯 {len(pairs)} paires de confusion × {args.n_per_pair} phrases = ~{len(pairs)*args.n_per_pair} phrases cibles")
    for p in pairs:
        print(f"   {p['id']:35s} ({p['target_label']} ↔ {', '.join(p['contrast_labels'])})")

    # Mode dry-run
    if args.dry_run:
        for p in pairs:
            print(f"\n{'='*70}")
            print(f"PAIRE : {p['id']}")
            print(f"{'='*70}")
            print(build_prompt(p, min(args.n_per_pair, 3)))
        return

    requests = build_batch_requests(pairs, args.n_per_pair, args.model)
    print(f"\n📦 {len(requests)} requêtes batch générées")

    if args.requests_file:
        Path(args.requests_file).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in requests))
        print(f"💾 Requêtes sauvegardées → {args.requests_file}")

    # Soumission ou reprise
    if args.batch_id:
        batch_id = args.batch_id
        print(f"♻️  Reprise batch : {batch_id}")
    else:
        batch_id = submit_batch(requests, api_key)

    # Polling
    print(f"\n⏳ Attente résultats (poll toutes les {args.poll_interval}s)...")
    results_url = poll_batch(batch_id, api_key, args.poll_interval)

    # Récupération
    print("\n📥 Récupération des résultats...")
    results = fetch_results(results_url, api_key)

    # Parsing et écriture
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_phrases = 0
    errors = 0
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            custom_id = result.get("custom_id", "?")
            if result.get("result", {}).get("type") == "error":
                print(f"  ❌ {custom_id}: {result['result']['error']}", file=sys.stderr)
                errors += 1
                continue

            content_blocks = result.get("result", {}).get("message", {}).get("content", [])
            content = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

            phrases = parse_claude_output(custom_id, content)
            for phrase in phrases:
                f.write(json.dumps(phrase, ensure_ascii=False) + "\n")
                total_phrases += 1

    print(f"\n✅ {total_phrases} phrases générées → {output_path}")
    print(f"   {errors} erreurs batch")
    if total_phrases > 0:
        print(f"\n📋 Étape suivante :")
        print(f"   python3 stanza_inject_svo_oblique_appos.py {output_path} -o data/contrastive_v1_svo.jsonl")
        print(f"   python3 scripts/review_stanza_spans_haiku_batch.py --input data/contrastive_v1_svo.jsonl --output data/contrastive_v1.jsonl")


if __name__ == "__main__":
    main()

