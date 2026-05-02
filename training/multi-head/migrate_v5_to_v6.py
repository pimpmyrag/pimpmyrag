"""
Migration v5 → v6 : ajout du label hint_inst_role

Règles de conversion :
  hint_group_role  → hint_inst_role  si le span désigne une institution générique
  hint_inst_name   → hint_inst_role  si le span est un terme générique sans qualificatif unique

Usage :
    python migrate_v5_to_v6.py [--dry-run]
"""
import json, collections, unicodedata, argparse, re
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Minuscule + suppression accents."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )


# ──────────────────────────────────────────────────────────────
# RÈGLES hint_group_role → hint_inst_role
# Un span group_role est converti si son texte normalisé COMMENCE
# par l'un de ces préfixes institutionnels (ou y est égal).
# ──────────────────────────────────────────────────────────────

GROUP_TO_INST_PREFIXES = [
    # État / exécutif
    'gouvernement',        # gouvernement, gouvernements, gouvernement français…
    'administration',
    'cabinet',             # cabinet ministériel
    'executif',
    # Législatif
    'parlement',           # parlement, parlements, parlement autrichien…
    'assemblee nationale',
    'assemblee generale',
    'senat',               # sénat, sénat américain
    'congres',             # congrès
    'chambre des lords',
    'chambre des deputes',
    'chambre haute',
    'chambre basse',
    'diete',               # Diète (parlement japonais/polonais)
    'bundestag',
    'bundesrat',
    'legislatif',
    # Judiciaire
    'tribunal',            # tribunal, tribunal correctionnel, tribunal administratif
    'cour d appel',
    'cour de cassation',
    'cour constitutionnelle',
    'cour supreme',
    'cour internationale',
    'cour de justice',
    'parquet',             # parquet (institution judiciaire)
    'judiciaire',
    # Forces de l'ordre / sécurité
    'police',              # police, police nationale, police locale…
    'gendarmerie',
    'forces de l ordre',
    'forces de police',
    'forces de securite',
    'forces armees',
    'forces aeriennes',
    'forces afghanes',
    'forces speciales',
    'forces internationales',
    'forces navales',
    'forces terrestres',
    'armee',               # armée, armée française, armée américaine…  (NB: "arme" = arme ≠ "armee")
    'marines',             # corps des marines
    'marine nationale',
    'marine americaine',
    'marine militaire',
    'marine royale',
    'prefecture de police',
    'prefecture',          # préfecture (représentant de l'État)
    'services de police',
    'autorites policières',
    'autorites',           # autorités, autorités locales, autorités sanitaires…
    'agents de securite',
    # Collectivités / administration locale
    'mairie',
    'municipalite',
    'conseil municipal',
    'conseil departemental',
    'conseil regional',
    'conseil general',
    'conseil des ministres',
    'conseil de securite',  # Conseil de sécurité (générique)
    # Organismes / commissions (sans nom propre)
    'commission',           # commission, commission nationale…  (différent de "Commission européenne" = inst_name)
    'comite',
    'syndicat',             # syndicat, syndicats
    'federation syndicale',
    'confederation',
    # Régime / coalition politique
    'coalition',
    'regime',               # régime politique
    # Service public
    'securite sociale',
    'service public',
    'services publics',
]

# Exceptions : ces préfixes provoquent des faux positifs → on ne convertit PAS
GROUP_BLACKLIST_PREFIXES = [
    'policier',    # personne
    'parlementaire',  # personne
    'senateur',    # personne
    'snateur',
    'congresiste',
    'conseiller',  # personne (conseiller municipal = hint_person_role)
    'ministre',    # personne (ministre = hint_person_role, pas ministère)
    'secour',      # "secours", "secouristes" ≠ institution
    'course',      # "tte de course" ≠ institution
    'coureur',     # cycliste
    'autoroute',   # infrastructure
    'autorisation',# acte admin, pas une institution
]


def should_convert_group_to_inst(text: str) -> bool:
    """True si le span hint_group_role doit devenir hint_inst_role."""
    n = normalize(text)
    # Blacklist d'abord
    for bl in GROUP_BLACKLIST_PREFIXES:
        if n.startswith(bl):
            return False
    # Vérifier les préfixes institutionnels
    for prefix in GROUP_TO_INST_PREFIXES:
        if n == prefix or n.startswith(prefix + ' ') or n.startswith(prefix + 's') \
                or n.startswith(prefix + 'x'):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# RÈGLES hint_inst_name → hint_inst_role
# Un inst_name est converti si c'est un terme générique
# (pas d'acronyme, pas d'adjectif géographique/propre fort)
# ──────────────────────────────────────────────────────────────

# Adjectifs/noms géographiques ou qualificatifs qui rendent le
# span suffisamment spécifique pour rester hint_inst_name
INST_NAME_QUALIFIERS = [
    'europeen', 'europeenne', 'europeens', 'europeennes',
    'francais', 'francaise', 'francs', 'france',
    'americain', 'americaine', 'americains',
    'britannique', 'anglais', 'anglaise',
    'allemand', 'allemande', 'allemands',
    'internationa',  # international/internationale
    'mondial', 'mondiale',
    'federal', 'federale', 'federaux',
    'supreme', 'suprema',
    'nations unies', 'onu ', ' onu',
    'europe', 'union europeenne',
    'de l onu', 'des nations',
    'de france', 'de paris',
    'de l europe',
    'constitutionnel', 'constitutionnelle',
    ' de ', ' des ', ' du ', ' d ',  # qualification par "de" → probablement spécifique
]

INST_NAME_GENERIC_ROOTS = [
    'gouvernement',
    'parlement',
    'senat',
    'congres',
    'assemblee',
    'tribunal',
    'parquet',
    'chancellerie',
    'gendarmerie',
    'prefecture',
    'administration',
    'executif',
    'legislatif',
    'judiciaire',
    'mairie',
    'syndicat',
    'coalition',
    'regime',
    'commission',
    'conseil des ministres',
    'conseil de securite',
    'conseil general',
    'conseil municipal',
    'conseil regional',
    'conseil departemental',
    'assemblee generale',
    'chambre des representants',
    'chambre haute',
    'chambre des deputes',
]


def should_convert_inst_name_to_role(text: str) -> bool:
    """True si le span hint_inst_name doit devenir hint_inst_role."""
    n = normalize(text)

    # Si le span contient un qualificatif géographique/institutionnel fort → reste inst_name
    for q in INST_NAME_QUALIFIERS:
        if q in n:
            return False

    # Si le texte est uniquement en majuscules (acronyme : ONU, OTAN…) → reste inst_name
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return False

    # Si un mot (autre que le premier) commence par une majuscule →
    # nom propre attaché (Bush, Fillon, Thatcher, Afghane…) → reste inst_name
    words = text.split()
    for w in words[1:]:
        # Ignore les articles/prépositions courants
        if w.lower() in ('de', 'du', 'des', 'le', 'la', 'les', 'et', 'sur', 'en',
                         "l'", 'un', 'une', 'à', 'au', 'par', 'd'):
            continue
        if w and w[0].isupper():
            return False

    # Vérifie si c'est un terme générique
    for root in INST_NAME_GENERIC_ROOTS:
        if n == root or n.startswith(root + ' ') or n.startswith(root + 's') \
                or n.startswith(root + 'x'):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# Traitement
# ──────────────────────────────────────────────────────────────

def migrate_file(src: Path, dst: Path, dry_run: bool) -> dict:
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    out_lines = []

    with open(src, encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            ex = json.loads(raw)
            changed = False
            for sp in ex.get('spans', []):
                lbl = sp['label']
                text = sp.get('text', '').strip()

                if lbl == 'hint_group_role' and should_convert_group_to_inst(text):
                    sp['label'] = 'hint_inst_role'
                    stats['group→inst_role'] += 1
                    if len(examples[f'group:{text}']) < 2:
                        examples[f'group:{text}'].append(text)
                    changed = True

                elif lbl == 'hint_inst_name' and should_convert_inst_name_to_role(text):
                    sp['label'] = 'hint_inst_role'
                    stats['inst_name→inst_role'] += 1
                    if len(examples[f'name:{text}']) < 2:
                        examples[f'name:{text}'].append(text)
                    changed = True

                elif lbl in ('hint_group_role', 'hint_inst_name', 'hint_inst_role'):
                    stats[f'kept_{lbl}'] += 1

            out_lines.append(json.dumps(ex, ensure_ascii=False))

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines) + '\n')

    return stats, examples


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

FILES = [
    ('data/train_v5.jsonl', 'data/train_v6.jsonl'),
    ('data/val_v5.jsonl',   'data/val_v6.jsonl'),
    ('data/test_v5.jsonl',  'data/test_v6.jsonl'),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Ne pas écrire les fichiers')
    args = parser.parse_args()

    base = Path(__file__).parent
    total_group = 0
    total_name  = 0

    for src_rel, dst_rel in FILES:
        src = base / src_rel
        dst = base / dst_rel
        print(f"\n{'='*60}")
        print(f"  {src.name}  →  {dst.name}")
        print(f"{'='*60}")
        stats, examples = migrate_file(src, dst, args.dry_run)

        g = stats.get('group→inst_role', 0)
        n = stats.get('inst_name→inst_role', 0)
        total_group += g
        total_name  += n

        print(f"  hint_group_role → hint_inst_role : {g:4d}")
        print(f"  hint_inst_name  → hint_inst_role : {n:4d}")
        print(f"  hint_inst_role résultant         : {g+n:4d}")
        print(f"  hint_group_role conservés        : {stats.get('kept_hint_group_role', 0):4d}")
        print(f"  hint_inst_name  conservés        : {stats.get('kept_hint_inst_name', 0):4d}")

        if g + n > 0:
            print(f"\n  Exemples convertis (group→inst_role) :")
            seen = set()
            for key, _ in sorted(examples.items()):
                if key.startswith('group:'):
                    label = key[6:]
                    if label not in seen:
                        seen.add(label)
                        print(f"    {label!r}")
                    if len(seen) > 20:
                        break
            print(f"\n  Exemples convertis (inst_name→inst_role) :")
            seen = set()
            for key, _ in sorted(examples.items()):
                if key.startswith('name:'):
                    label = key[5:]
                    if label not in seen:
                        seen.add(label)
                        print(f"    {label!r}")

        if args.dry_run:
            print(f"\n  [DRY-RUN] Fichier {dst.name} non écrit.")
        else:
            n_lines = sum(1 for _ in open(dst))
            print(f"\n  ✅ Écrit : {dst.name}  ({n_lines} lignes)")

    print(f"\n{'='*60}")
    print(f"  TOTAL group→inst_role : {total_group}")
    print(f"  TOTAL name →inst_role : {total_name}")
    print(f"  TOTAL hint_inst_role  : {total_group + total_name}")
    print(f"{'='*60}")
    if args.dry_run:
        print("\n  ⚠️  Mode DRY-RUN : aucun fichier modifié.")
    else:
        print("\n  ✅ Migration terminée. Prochaine étape : dvc add + dvc push")

