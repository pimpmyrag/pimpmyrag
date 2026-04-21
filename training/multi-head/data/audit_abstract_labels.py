#!/usr/bin/env python3
"""Audit du dataset existant pour trouver les spans mal labellisés
qui devraient être reannotés avec les nouveaux labels ABSTRACT."""
import json
from collections import Counter

law_kw = ['traité', 'édit', 'loi ', 'décret', 'constitution', 'convention',
           'accord', 'pacte', 'charte', 'code civil', 'protocole',
           'directive', 'ordonnance', 'concordat', 'règlement', 'armistice',
           'amnistie', 'bill', 'acte']
disease_kw = ['maladie', 'virus', 'grippe', 'peste', 'choléra', 'cancer',
              'diabète', 'covid', 'épidémie', 'tuberculose', 'sida', 'vih',
              'ebola', 'variole', 'paludisme', 'fièvre', 'syndrome',
              'infection', 'rougeole', 'dengue', 'rage', 'asthme', 'zika',
              'sras', 'mpox', 'polio']
concept_kw = ['théorie', 'doctrine', 'idéologie', 'marxisme', 'capitalisme',
              'socialisme', 'libéralisme', 'féminisme', 'communisme',
              'fascisme', 'anarchisme', 'darwinisme', 'relativité',
              'démocratie', 'laïcité', 'psychanalyse', 'existentialisme',
              'positivisme', 'empirisme', 'nihilisme', 'stoïcisme',
              'rationalisme', 'humanisme', 'surréalisme', 'réalisme',
              'romantisme', 'cubisme', 'impressionnisme', '-isme']
language_kw = ['français', 'anglais', 'espagnol', 'allemand', 'arabe',
               'mandarin', 'latin', 'grec', 'japonais', 'russe', 'chinois',
               'portugais', 'italien', 'hébreu', 'swahili', 'hindi',
               'persan', 'turc', 'coréen']

categories = {
    'hint_law': law_kw,
    'hint_disease': disease_kw,
    'hint_concept': concept_kw,
    'hint_language': language_kw,
}

results = {cat: [] for cat in categories}

for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    with open(f'data/{split}', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            for sp in row.get('spans', []):
                txt = sp['text'].lower()
                lab = sp['label']
                if lab.startswith('hint_law') or lab.startswith('hint_disease') or \
                   lab.startswith('hint_concept') or lab.startswith('hint_language') or \
                   lab.startswith('hint_work_of_art'):
                    continue
                for cat, keywords in categories.items():
                    for kw in keywords:
                        if kw in txt:
                            results[cat].append({
                                'split': split,
                                'current_label': lab,
                                'text': sp['text'][:80],
                                'sentence': row['text'][:120],
                                'id': row['id'],
                            })
                            break

for cat, items in results.items():
    print(f"\n{'='*70}")
    print(f"  {cat.upper()} suspects: {len(items)} spans")
    print(f"{'='*70}")
    label_counts = Counter(it['current_label'] for it in items)
    for lab, cnt in label_counts.most_common():
        print(f"  {lab:30s} {cnt:4d}")
    print(f"\n  Exemples:")
    for it in items[:15]:
        print(f"    [{it['split']:12s}] {it['current_label']:25s} -> \"{it['text']}\"")
        print(f"                  sentence: \"{it['sentence']}\"")

total = sum(len(v) for v in results.values())
print(f"\n\n{'='*70}")
print(f"  TOTAL suspects: {total}")
print(f"{'='*70}")

