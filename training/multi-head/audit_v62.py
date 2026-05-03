"""
Audit cohérence des corrections Mistral dans v6.2
"""
import json
from collections import Counter, defaultdict

corrections = {}
all_results = []
with open('data/mistral_batch_review.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        all_results.append(r)
        if r['verdict'] == 'SUSPECT' and r.get('label_suggested') and r['label_suggested'] != r['label']:
            if r['label_suggested'] not in ('non-NER', 'hint_norp', 'hint_other'):
                corrections[(r['label'], r['span'])] = (r['label_suggested'], r.get('raison', ''))

# 1. Contradictions : même span_text corrigé en directions opposées
print("=== 1. CONTRADICTIONS (même texte, labels croisés) ===")
by_span = defaultdict(list)
for (src, span), (dst, _) in corrections.items():
    by_span[span.lower()].append((src, dst))
found = 0
for span, moves in sorted(by_span.items()):
    if len(moves) > 1:
        print(f'  "{span}": {moves}')
        found += 1
print(f"  Total: {found}\n")

# 2. org_name→inst_name très courts (<=8 cars) ou très génériques
print("=== 2. org_name→inst_name SUSPECTS (courts ou génériques) ===")
for (src, span), (dst, raison) in sorted(corrections.items()):
    if src == 'hint_org_name' and dst == 'hint_inst_name' and len(span) <= 8:
        print(f'  "{span}" ({len(span)} cars) | {raison[:90]}')
print()

# 3. group_role→inst_name douteux (mots génériques)
print("=== 3. group_role→inst_name potentiellement trop agressif ===")
generic = ['justice', 'concile', 'délégation', 'syndicat', 'fédération', 'comité', 'commission', 'section']
for (src, span), (dst, raison) in sorted(corrections.items()):
    if src == 'hint_group_role' and dst == 'hint_inst_name':
        if any(g in span.lower() for g in generic) or len(span) < 12:
            print(f'  "{span}" | {raison[:90]}')
print()

# 4. org_name gardés OK qui ressemblent à inst
print("=== 4. org_name restés OK mais vocabulaire institutionnel ===")
inst_words = ['ministère', 'parlement', 'assemblée', 'conseil', 'cour', 'tribunal',
              'police', 'armée', 'marine', 'gendarmerie', 'préfecture', 'gouvernement']
ok_inst = []
for r in all_results:
    if r['label'] == 'hint_org_name' and r['verdict'] == 'OK':
        if any(w in r['span'].lower() for w in inst_words):
            ok_inst.append(r['span'])
for s in sorted(ok_inst):
    print(f'  "{s}"')
print()

# 5. inst_role→inst_name : tous les cas (sont-ils vraiment nommés ?)
print("=== 5. inst_role→inst_name: liste complète ===")
for (src, span), (dst, raison) in sorted(corrections.items()):
    if src == 'hint_inst_role' and dst == 'hint_inst_name':
        print(f'  "{span}" | {raison[:90]}')
print()

# 6. Cohérence interne : spans identiques avec verdicts différents selon label
print("=== 6. Même span_text, verdicts CONTRADICTOIRES (OK vs SUSPECT) ===")
by_text = defaultdict(list)
for r in all_results:
    by_text[r['span'].lower()].append((r['label'], r['verdict'], r.get('label_suggested', '')))
found2 = 0
for span, entries in sorted(by_text.items()):
    verdicts = set(e[1] for e in entries)
    if 'OK' in verdicts and 'SUSPECT' in verdicts:
        print(f'  "{span}":')
        for lbl, verd, sugg in entries:
            print(f'    [{lbl}] {verd}' + (f' → {sugg}' if sugg and sugg != lbl else ''))
        found2 += 1
        if found2 >= 20:
            print('  ... (tronqué à 20)')
            break
print(f"  Total: {found2}")

