#!/usr/bin/env python3
"""
Télécharge des articles Wikipedia FR par catégorie ciblée pour enrichir
le dataset d'entraînement NER sur les labels faibles.

Labels cibles → catégories Wikipedia FR :
  hint_food        → Cuisine française, Gastronomie, Aliment, Fruit, Légume...
  hint_doctrine    → Philosophie politique, Idéologie, Courant philosophique...
  hint_state       → Régime politique, Droit constitutionnel, Science politique...
  hint_notion      → Concept philosophique, Épistémologie, Logique...
  hint_weapon      → Arme, Armement, Arme à feu, Explosif, Véhicule militaire...
  hint_inst_name   → Institution française, Organisation internationale...
  hint_work_generic→ Mouvement artistique, Courant littéraire, Genre musical...
  hint_object_generic → Outil, Ustensile, Matériau, Objet du quotidien...
  hint_event_named → Révolution, Guerre, Traité, Attentat, Sommet...
  hint_inst_role   → Fonction publique, Grade militaire, Titre de noblesse...
  hint_substance   → Médicament, Minéral, Composé chimique, Textile...

Usage :
    python3 download_fr_wiki_categories.py --output data/frwiki_silver_raw.jsonl
"""
import argparse
import json
import time
import re
import urllib.request
import urllib.parse
from pathlib import Path

API_URL = "https://fr.wikipedia.org/w/api.php"

TARGET_CATEGORIES = {
    "hint_food": [
        "Cuisine française",
        "Gastronomie",
        "Boisson",
        "Fruit",
        "Légume",
        "Fromage",
        "Sauce",
        "Pâtisserie",
        "Viande",
        "Épice",
        "Cuisine du monde",
        "Alimentation",
    ],
    "hint_doctrine": [
        "Idéologie politique",
        "Philosophie politique",
        "Courant philosophique",
        "Courant de pensée",
        "Théologie",
        "École de pensée économique",
        "Doctrine militaire",
    ],
    "hint_state": [
        "Concept politique",
        "Régime politique",
        "Droit constitutionnel",
        "Science politique",
        "Droit international",
        "Forme de gouvernement",
        "Système politique",
    ],
    "hint_notion": [
        "Concept philosophique",
        "Épistémologie",
        "Philosophie de l'esprit",
        "Logique",
        "Ontologie",
        "Concept sociologique",
        "Concept économique",
        "Concept juridique",
    ],
    "hint_weapon": [
        "Arme",
        "Arme blanche",
        "Arme à feu",
        "Explosif",
        "Véhicule militaire",
        "Missile",
        "Arme de destruction massive",
        "Équipement militaire",
    ],
    "hint_inst_name": [
        "Institution française",
        "Organisation internationale",
        "Institution européenne",
        "Tribunal",
        "Agence gouvernementale",
        "Organisation non gouvernementale",
        "Institution financière internationale",
        "Parlement",
    ],
    "hint_work_generic": [
        "Mouvement artistique",
        "Courant littéraire",
        "Genre musical",
        "Style architectural",
        "Genre cinématographique",
        "Courant artistique",
        "Avant-garde artistique",
    ],
    "hint_object_generic": [
        "Outil",
        "Ustensile de cuisine",
        "Matériau de construction",
        "Instrument de musique",
        "Instrument scientifique",
        "Équipement sportif",
        "Mobilier",
        "Objet liturgique",
    ],
    "hint_event_named": [
        "Révolution",
        "Guerre",
        "Traité international",
        "Attentat terroriste",
        "Sommet international",
        "Coup d'État",
        "Soulèvement populaire",
        "Crise politique",
    ],
    "hint_inst_role": [
        "Fonction publique en France",
        "Grade militaire",
        "Titre de noblesse",
        "Profession juridique",
        "Titre honorifique",
        "Charge ecclésiastique",
        "Fonction diplomatique",
    ],
    "hint_substance": [
        "Médicament",
        "Minéral",
        "Composé chimique",
        "Matière plastique",
        "Textile",
        "Métal",
        "Matériau composite",
        "Substance psychoactive",
    ],
}

MAX_ARTICLES_PER_CATEGORY = 200
MIN_TEXT_LENGTH = 200  # caractères minimum par phrase
MAX_SENTENCES_PER_ARTICLE = 20


HEADERS = {
    "User-Agent": "pimpmyrag-ner-dataset/1.0 (research; contact@pimpmyrag.com) urllib/3",
    "Accept": "application/json",
}

def api_get(params: dict) -> dict:
    params["format"] = "json"
    params["action"] = params.get("action", "query")
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_category_members(category: str, limit: int = MAX_ARTICLES_PER_CATEGORY) -> list[str]:
    """Récupère les titres d'articles d'une catégorie Wikipedia."""
    titles = []
    params = {
        "list": "categorymembers",
        "cmtitle": f"Catégorie:{category}",
        "cmlimit": min(limit, 500),
        "cmtype": "page",
    }
    while len(titles) < limit:
        data = api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont or len(titles) >= limit:
            break
        params["cmcontinue"] = cont
        time.sleep(0.1)
    return titles[:limit]


def get_article_text(title: str) -> str | None:
    """Récupère le texte brut d'un article Wikipedia."""
    params = {
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "titles": title,
        "redirects": True,
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if text and len(text) > 100:
            return text
    return None


def split_sentences(text: str) -> list[str]:
    """Split basique en phrases françaises."""
    # Supprime les sections == Titre == et les listes
    text = re.sub(r"==+[^=]+=+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Split sur . ! ? suivi d'une majuscule
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ])", text)
    # Filtrer les phrases trop courtes ou trop longues
    return [
        s.strip() for s in sentences
        if MIN_TEXT_LENGTH // 4 < len(s.strip()) < 1000
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/frwiki_silver_raw.jsonl")
    parser.add_argument("--labels", nargs="+", default=list(TARGET_CATEGORIES.keys()),
                        help="Labels à cibler (défaut: tous)")
    parser.add_argument("--max-articles", type=int, default=MAX_ARTICLES_PER_CATEGORY)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    stats = {}

    with open(output_path, "w", encoding="utf-8") as fout:
        for label in args.labels:
            categories = TARGET_CATEGORIES.get(label, [])
            label_sentences = 0
            seen_titles = set()

            print(f"\n{'='*60}")
            print(f"🏷️  Label : {label}  ({len(categories)} catégories)")

            for category in categories:
                print(f"  📂 Catégorie : {category}", flush=True)
                try:
                    titles = get_category_members(category, args.max_articles)
                    print(f"     → {len(titles)} articles trouvés")
                except Exception as e:
                    print(f"     ⚠️  Erreur: {e}")
                    continue

                for i, title in enumerate(titles):
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    try:
                        text = get_article_text(title)
                        if not text:
                            continue
                        sentences = split_sentences(text)[:MAX_SENTENCES_PER_ARTICLE]
                        for sent in sentences:
                            record = {
                                "text": sent,
                                "source": "frwiki",
                                "category": category,
                                "target_label": label,
                                "article": title,
                                "annotated": False,
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            label_sentences += 1
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"     ⚠️  {title}: {e}")
                        continue

                    if (i + 1) % 20 == 0:
                        print(f"     ... {i+1}/{len(titles)} articles, {label_sentences} phrases", flush=True)

            stats[label] = label_sentences
            total_sentences += label_sentences
            print(f"  ✅ {label} : {label_sentences} phrases extraites")

    print(f"\n{'='*60}")
    print(f"✅ Terminé — {total_sentences} phrases au total → {output_path}")
    print(f"\nPar label :")
    for label, n in stats.items():
        print(f"  {label:<25} {n:>6} phrases")


if __name__ == "__main__":
    main()

