"""
Envoie les spans group_role / inst_name / inst_role / org_name à Mistral Large
par batches de 30 pour verdict OK / SUSPECT.

Format réponse attendu par batch:
[{"id": 1, "verdict": "OK"},
 {"id": 2, "verdict": "SUSPECT", "label": "hint_inst_role", "raison": "..."},
 ...]

Sortie: data/mistral_batch_review.jsonl  (une ligne par paire unique)
Résumé TSV: data/org_spans_review_with_verdict.tsv
"""
import json, time, re, requests
from pathlib import Path
from collections import defaultdict

API_KEY  = "sX7fpLMqKqXbHx5bRtxTZSkjK0EWfdnF"
API_URL  = "https://api.mistral.ai/v1/chat/completions"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

BATCH_SIZE   = 30
SLEEP_SEC    = 1.2   # >1s pour éviter 429
RESULTS_FILE = Path("data/mistral_batch_review.jsonl")

TARGET = {'hint_group_role', 'hint_inst_name', 'hint_inst_role', 'hint_org_name'}
LABELS_DOC = (
    "hint_group_role = groupe de personnes défini par rôle/caractéristique (manifestants, syndicats de médecins, rebelles...)\n"
    "hint_inst_name  = institution NOMMÉE avec qualificatif résolvant : géographique, NORP, nom propre (police de Paris, gouvernement français, ONU)\n"
    "hint_inst_role  = institution GÉNÉRIQUE sans qualificatif résolvant (le gouvernement, la police, le tribunal)\n"
    "hint_org_name   = organisation nommée : entreprise, parti, ONG, média (Apple, PS, Médecins sans frontières, Le Monde)"
)

SYSTEM_PROMPT = f"""Tu es expert en annotation NER pour le français.
Règles de classification:
{LABELS_DOC}

Pour chaque span numéroté, réponds UNIQUEMENT avec un tableau JSON compact:
- si le label est correct: {{"id":N,"verdict":"OK"}}
- si suspect: {{"id":N,"verdict":"SUSPECT","label":"hint_xxx","raison":"courte raison"}}
Ne génère rien d'autre que le tableau JSON."""

# ── 1. Charge les paires uniques ────────────────────────────────────────────
unique = defaultdict(list)   # (label, span_text) -> [ctx1, ctx2]

for split in ['train', 'val', 'test']:
    p = Path(f'data/{split}_v6.1.jsonl')
    if not p.exists():
        continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            text = d['text']
            for span in d.get('spans', []):
                if span['label'] not in TARGET:
                    continue
                key = (span['label'], span.get('text', text[span['start']:span['end']]))
                if len(unique[key]) < 2:
                    s, e = span['start'], span['end']
                    ctx = text[max(0,s-55):s] + '[[' + text[s:e] + ']]' + text[e:min(len(text),e+55)]
                    unique[key].append(ctx.replace('\n', ' '))

items = [{'label': k[0], 'span': k[1], 'contexts': v} for k, v in unique.items()]
print(f"{len(items)} paires uniques  |  {(len(items)+BATCH_SIZE-1)//BATCH_SIZE} batches")

# ── 2. Charge résultats déjà faits (reprise) ────────────────────────────────
done = {}   # (label, span) -> résultat
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            done[(r['label'], r['span'])] = r
    print(f"Reprise: {len(done)} déjà traités")

# ── 3. Filtre ce qui reste ───────────────────────────────────────────────────
todo = [it for it in items if (it['label'], it['span']) not in done]
print(f"Reste à traiter: {len(todo)}")

# ── 4. Envoi par batches ─────────────────────────────────────────────────────
def call_mistral(user_msg: str) -> str:
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        "temperature": 0.0
    }
    for attempt in range(5):
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 429:
            wait = 5 * (2 ** attempt)
            print(f"  429 → attente {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    raise RuntimeError("5 tentatives épuisées (429)")

def parse_response(response: str, batch: list) -> list:
    """Parse le JSON renvoyé par Mistral, retourne une liste de résultats."""
    # Extrait le tableau JSON (peut être enveloppé dans ```json ... ```)
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        raise ValueError(f"Pas de tableau JSON dans: {response[:200]}")
    verdicts = json.loads(match.group())
    verdicts_by_id = {v['id']: v for v in verdicts}
    results = []
    for i, item in enumerate(batch, start=1):
        v = verdicts_by_id.get(i, {'id': i, 'verdict': 'PARSE_MISSING'})
        results.append({
            'label':   item['label'],
            'span':    item['span'],
            'verdict': v.get('verdict', 'PARSE_MISSING'),
            'label_suggested': v.get('label'),
            'raison':  v.get('raison'),
        })
    return results

success = errors = 0
batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

with open(RESULTS_FILE, 'a', encoding='utf-8') as fout:
    for bi, batch in enumerate(batches):
        # Construit le message utilisateur
        lines = []
        for i, item in enumerate(batch, start=1):
            ctx_str = ' | '.join(item['contexts'])
            lines.append(f"{i}. [{item['label']}] \"{item['span']}\" — {ctx_str}")
        user_msg = "Évalue ces spans:\n" + "\n".join(lines)

        print(f"[batch {bi+1}/{len(batches)}] {len(batch)} spans...", end=' ', flush=True)
        try:
            response = call_mistral(user_msg)
            results  = parse_response(response, batch)
            suspects = sum(1 for r in results if r['verdict'] != 'OK')
            print(f"OK ({suspects} suspects)")
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            success += len(results)
        except Exception as e:
            print(f"ERREUR: {e}")
            for item in batch:
                r = {'label': item['label'], 'span': item['span'], 'verdict': 'ERROR', 'error': str(e)}
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            errors += len(batch)

        time.sleep(SLEEP_SEC)

print(f"\n✅ {success} traités, {errors} erreurs")

# ── 5. Génère TSV final avec verdicts ────────────────────────────────────────
print("Génération du TSV final...")
verdicts = {}
with open(RESULTS_FILE, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        verdicts[(r['label'], r['span'])] = r

# Recharge le TSV source
import csv
tsv_in  = Path("data/org_spans_review.tsv")
tsv_out = Path("data/org_spans_review_with_verdict.tsv")

with open(tsv_in, encoding='utf-8', newline='') as fin, \
     open(tsv_out, 'w', encoding='utf-8', newline='') as fout:
    reader = csv.DictReader(fin, delimiter='\t')
    fieldnames = reader.fieldnames + ['verdict', 'label_suggested', 'raison']
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for row in reader:
        key = (row['current_label'], row['span_text'])
        v = verdicts.get(key, {})
        row['verdict']         = v.get('verdict', '')
        row['label_suggested'] = v.get('label_suggested') or ''
        row['raison']          = v.get('raison') or ''
        writer.writerow(row)

# Stats
suspects_count = sum(1 for v in verdicts.values() if v.get('verdict') == 'SUSPECT')
ok_count       = sum(1 for v in verdicts.values() if v.get('verdict') == 'OK')
print(f"TSV final: {tsv_out}")
print(f"  OK: {ok_count}  SUSPECT: {suspects_count}  autres: {len(verdicts)-ok_count-suspects_count}")

# Top suspects par label
from collections import Counter
print("\nTop 20 SUSPECTS:")
suspects_list = [(v['label'], v['span'], v.get('label_suggested','?'), v.get('raison','')[:60])
                 for v in verdicts.values() if v.get('verdict') == 'SUSPECT']
suspects_list.sort()
for lbl, span, sugg, raison in suspects_list[:20]:
    print(f"  [{lbl}] \"{span}\" → {sugg} | {raison}")

