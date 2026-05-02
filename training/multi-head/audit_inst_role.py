"""
Audit des hint_group_role qui sont en realite des institutions generiques
et pourraient devenir hint_inst_role.
"""
import json, collections

INST_KEYWORDS = [
    'gouvernement', 'parlement', 'senat', 'congres', 'assemblee',
    'conseil', 'tribunal', 'cour', 'parquet', 'chancellerie',
    'gendarmerie', 'police', 'armee', 'forces', 'marine',
    'ministere', 'ministre', 'administration', 'etat', 'prefecture',
    'mairie', 'municipalite', 'chambre', 'bundestag',
    'comite', 'commission', 'autorite', 'agence', 'office',
    'syndicat', 'federation', 'confederation',
    'coalition', 'regime', 'autorites',
    'cabinet', 'executif', 'legislatif', 'judiciaire',
]

def normalize(s):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )

counter_group = collections.Counter()
examples = collections.defaultdict(list)

with open('data/train_v5.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        text = ex.get('text', '')
        for sp in ex.get('spans', []):
            if sp['label'] != 'hint_group_role':
                continue
            t_norm = normalize(sp['text'].strip())
            for kw in INST_KEYWORDS:
                if kw in t_norm:
                    key = sp['text'].strip()
                    counter_group[key] += 1
                    if len(examples[key]) < 2:
                        s, e = sp['start'], sp['end']
                        ctx = text[max(0, s-35):e+35]
                        examples[key].append(ctx)
                    break

print(f"hint_group_role avec mot-cle institutionnel : {sum(counter_group.values())} spans, {len(counter_group)} valeurs distinctes")
print()
for text_key, count in sorted(counter_group.items(), key=lambda x: -x[1])[:80]:
    print(f"  [{count:3d}x]  {text_key!r}")
    for ctx in examples[text_key]:
        print(f"           -> ...{ctx}...")

