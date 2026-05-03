"""
Filtre strict des corrections Mistral → v6.3

Règles :
1. Exclure les équipes sportives annotées comme inst_name
2. Exclure les spans non-institutions (SMIC, CPE, AOP...)
3. Pour org_name→inst_name spans courts (<10 cars) : garder seulement
   ceux dont la raison mentionne explicitement un qualificatif résolvant
4. Supprimer les corrections contradictoires (même texte → destinations différentes)
5. Pour spans avec contradictions OK/SUSPECT entre batches : garder si
   la majorité des occurrences suggère le même changement
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path

# ── Listes d'exclusion ────────────────────────────────────────────────────────
SPORTS_CITIES = {
    'agen', 'lyon', 'amiens', 'le havre', 'toulouse', 'madrid', 'limoges',
    'nantes', 'bordeaux', 'lille', 'marseille', 'monaco', 'nice', 'rennes',
    'reims', 'lens', 'metz', 'strasbourg', 'brest', 'montpellier', 'angers',
    'troyes', 'lorient', 'caen', 'grenoble', 'nancy', 'ajaccio', 'bastia',
    'sochaux', 'valenciennes', 'tours', 'laval', 'gueugnon', 'sedan',
    'le mans', 'boks', 'all blacks', 'canaris', 'dynamiques'
}

NON_INSTITUTIONS = {
    'smic', 'cpe', 'aop', 'wts', 'zdc', 'zec', 'crds', 'csg',
    'admin.ch', 'presse', 'ligue', 'officine', 'muse', 'fac',
    'agriculture biologique', 'annuaire statistique', 'mittani',
    'ferme',  # métonymie trop indirecte
}

# Mots qui indiquent un qualificatif résolvant dans la raison
RESOLVING_KEYWORDS = [
    'géographique', 'gographique', 'norp', 'qualificatif', 'résolvant', 'rsolvant',
    'acronyme', 'sigle connu', 'universellement', 'officiell', 'spcifique',
    'institution nomm', 'nommée', 'nomm'
]

def has_resolving_qualifier(raison: str) -> bool:
    raison_lower = raison.lower()
    return any(kw.lower() in raison_lower for kw in RESOLVING_KEYWORDS)

# ── Charge toutes les corrections brutes ─────────────────────────────────────
all_results = []
with open('data/mistral_batch_review.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        all_results.append(r)

# ── Groupe par (label, span_lower) pour détecter contradictions ──────────────
# Vote majoritaire parmi les SUSPECT avec label_suggested différent
group = defaultdict(list)  # (label, span_lower) -> liste de label_suggested
for r in all_results:
    if r['verdict'] == 'SUSPECT' and r.get('label_suggested'):
        if r['label_suggested'] not in ('non-NER', 'hint_norp', 'hint_other'):
            if r['label_suggested'] != r['label']:
                group[(r['label'], r['span'].lower())].append(
                    (r['label_suggested'], r.get('raison', ''), r['span'])
                )

# ── Construire la table de corrections filtrée ────────────────────────────────
corrections = {}  # (label, span_text_original) -> new_label
stats_excluded = Counter()
stats_kept = Counter()

for (src_label, span_lower), suggestions in group.items():
    # Récupère le span original (casse d'origine)
    span_orig = suggestions[0][2]

    # Exclure équipes sportives
    if span_lower in SPORTS_CITIES:
        stats_excluded['sports_team'] += 1
        continue

    # Exclure non-institutions
    if span_lower in NON_INSTITUTIONS:
        stats_excluded['non_institution'] += 1
        continue

    # Vote majoritaire sur le label suggéré
    vote = Counter(s[0] for s in suggestions)
    best_dst, best_count = vote.most_common(1)[0]

    # Contradiction : plusieurs destinations différentes
    if len(vote) > 1 and best_count == 1:
        stats_excluded['contradiction'] += 1
        continue

    # Pour org_name → inst_name sur spans très courts : vérifier la raison
    if src_label == 'hint_org_name' and best_dst == 'hint_inst_name' and len(span_lower) <= 8:
        # Accepter seulement si la raison mentionne un qualificatif résolvant
        best_raison = next(s[1] for s in suggestions if s[0] == best_dst)
        if not has_resolving_qualifier(best_raison):
            stats_excluded['short_no_qualifier'] += 1
            continue

    corrections[(src_label, span_orig)] = best_dst
    stats_kept[(src_label, best_dst)] += 1

print(f"Corrections filtrées : {len(corrections)}")
print(f"\nGardées:")
for k, n in sorted(stats_kept.items()):
    print(f"  {k[0]} → {k[1]}: {n}")
print(f"\nExclues:")
for reason, n in sorted(stats_excluded.items()):
    print(f"  {reason}: {n}")

# ── Applique au dataset v6.1 → v6.3 ─────────────────────────────────────────
print("\n--- Application au dataset ---")
for split in ['train', 'val', 'test']:
    in_path = Path(f'data/{split}_v6.1.jsonl')
    out_path = Path(f'data/{split}_v6.3.jsonl')
    if not in_path.exists():
        print(f'SKIP {in_path}')
        continue

    changed_spans = changed_sents = total = 0
    with open(in_path, encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            modified = False
            for span in d.get('spans', []):
                key = (span['label'], span.get('text', ''))
                if key in corrections:
                    span['label'] = corrections[key]
                    changed_spans += 1
                    modified = True
            if modified:
                changed_sents += 1
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f'[{split}] {total} phrases | {changed_sents} modifiées | {changed_spans} spans → {out_path}')

# ── Distribution finale ───────────────────────────────────────────────────────
print('\n=== Distribution labels ORG (train) ===')
TARGET = {'hint_group_role', 'hint_inst_name', 'hint_inst_role', 'hint_org_name'}
for version, fname in [('v6.1', 'train_v6.1.jsonl'), ('v6.3', 'train_v6.3.jsonl')]:
    p = Path(f'data/{fname}')
    if not p.exists():
        continue
    c = Counter()
    with open(p, encoding='utf-8') as f:
        for line in f:
            for span in json.loads(line).get('spans', []):
                if span['label'] in TARGET:
                    c[span['label']] += 1
    print(f'\n{version}:')
    for lbl, n in sorted(c.items()):
        print(f'  {lbl}: {n}')

