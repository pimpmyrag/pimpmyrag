#!/usr/bin/env python3
"""
Extrait les articles du dump WikiNews FR (XML bz2) en phrases JSONL.
Sortie : une ligne par phrase, format {"id": "wn_000001", "text": "..."}
"""
import argparse
import bz2
import json
import re
import xml.etree.ElementTree as ET


def strip_wikimarkup(text: str) -> str:
    """Nettoyage basique du wikitext → texte brut."""
    # Supprimer les balises HTML
    text = re.sub(r"<[^>]+>", "", text)
    # [[Lien|texte affiché]] → texte affiché
    text = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    # [[Lien simple]] → Lien simple
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # {{modèle|...}} → supprimer
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Liens externes [http://... texte] → texte
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # Gras/italique
    text = re.sub(r"'{2,5}", "", text)
    # Titres == ... ==
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1.", text, flags=re.MULTILINE)
    # Listes * / # / ;
    text = re.sub(r"^[*#;:]+\s*", "", text, flags=re.MULTILINE)
    # Tables {| ... |}
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # Lignes de table
    text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^!.*$", "", text, flags=re.MULTILINE)
    # Catégories [[Catégorie:...]]
    text = re.sub(r"\[\[Catégorie:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    # Interwiki [[xx:...]]
    text = re.sub(r"\[\[[a-z]{2,3}:[^\]]*\]\]", "", text)
    # Nettoyage espaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split naïf en phrases (sur . ! ? suivi d'une majuscule ou fin)."""
    # Split sur ponctuation forte suivie d'espace + majuscule
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖÙ-Ü0-9«"])', text)
    sentences = []
    for p in parts:
        p = p.strip()
        if len(p) >= 20 and len(p) <= 500:  # phrases de taille raisonnable
            sentences.append(p)
    return sentences


SKIP_TITLES = {
    "Wikinews:","Catégorie:","Modèle:","Utilisateur:",
    "Discussion:","Aide:","Portail:","Fichier:","MediaWiki:",
}


def parse_wikinews_dump(bz2_path: str):
    """Générateur qui yield (title, text_brut) pour chaque article."""
    print(f"📖 Lecture de {bz2_path} ...")
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        # Itérer sur les événements XML pour économiser la RAM
        context = ET.iterparse(f, events=("end",))
        ns = "{http://www.mediawiki.org/xml/export-0.11/}"

        for event, elem in context:
            if elem.tag == f"{ns}page":
                title_el = elem.find(f"{ns}title")
                text_el = elem.find(f".//{ns}text")

                title = title_el.text if title_el is not None else ""
                raw = text_el.text if text_el is not None else ""

                # Skip pages spéciales
                if any(title.startswith(s) for s in SKIP_TITLES):
                    elem.clear()
                    continue

                # Skip redirections
                if raw and raw.strip().upper().startswith("#REDIRECT"):
                    elem.clear()
                    continue

                if raw and len(raw) > 100:
                    yield title, raw

                elem.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin vers le .xml.bz2")
    parser.add_argument("--output", required=True, help="Chemin de sortie .jsonl")
    parser.add_argument("--max-articles", type=int, default=None)
    args = parser.parse_args()

    n_articles = 0
    n_sentences = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for title, raw in parse_wikinews_dump(args.input):
            clean = strip_wikimarkup(raw)
            sentences = split_sentences(clean)

            for sent in sentences:
                n_sentences += 1
                record = {
                    "id": f"wn_{n_sentences:06d}",
                    "text": sent,
                    "source_title": title,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_articles += 1
            if n_articles % 1000 == 0:
                print(f"  {n_articles} articles → {n_sentences} phrases")

            if args.max_articles and n_articles >= args.max_articles:
                break

    print(f"\n✅ {n_articles} articles → {n_sentences} phrases extraites → {args.output}")


if __name__ == "__main__":
    main()

