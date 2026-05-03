"""
Révision Claude Haiku des labels INSTITUTION du dataset v6.4.

Labels analysés : hint_inst_name, hint_inst_role, hint_org_name

Frontière clé :
  - inst_name  : institution PUBLIQUE avec nom propre/sigle (ONU, Sénat américain, BCE)
  - inst_role  : institution PUBLIQUE générique sans nom propre (gouvernement, police, armée)
  - org_name   : organisation PRIVÉE/civile avec nom propre (Apple, CGT, Le Monde, PSG)

Sortie : data/inst_labels_review.jsonl
"""
import json, time, re, os, requests
from pathlib import Path
from collections import defaultdict

# ── Config API ─────────────────────────────────────────────────────────────────
for line in open(Path(__file__).parent / '.secrets.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY  = os.environ["ANTHROPIC_API_KEY"]
API_URL  = "https://api.anthropic.com/v1/messages"
HEADERS  = {
    "x-api-key":         API_KEY,
    "anthropic-version": "2023-06-01",
    "Content-Type":      "application/json",
}

BATCH_SIZE   = 10
SLEEP_SEC    = 10.0
RESULTS_FILE = Path("data/inst_labels_review.jsonl")

TARGET = {'hint_inst_name', 'hint_inst_role', 'hint_org_name'}

SYSTEM_PROMPT = """Tu es expert en annotation NER pour le français, appliqué à l'extraction d'événements.

PHILOSOPHIE : garder au MAXIMUM les annotations. Ne suggère REMOVE que pour les vrais bruits manifestes
(fragment incomplet, déterminant seul, contenu illisible).

Tu dois distinguer 3 labels d'organisations/institutions :

  hint_org_name   : Organisation PRIVÉE ou CIVILE avec nom propre :
                    - entreprise ("Apple", "TotalEnergies", "Renault", "RTE")
                    - parti politique nommé ("le PS", "En Marche", "le FN", "les Républicains")
                    - syndicat nommé ("la CGT", "FO", "la CFDT")
                    - association / ONG ("la Croix-Rouge", "Amnesty International")
                    - média nommé ("Le Monde", "BFMTV", "France 2", "Reuters")
                    - club sportif ("le PSG", "l'OM", "le Real Madrid")
                    - organisation internationale PRIVÉE ou mixte sans mandat étatique direct
                    NON : institution publique/étatique → hint_inst_name ou hint_inst_role.

  hint_inst_name  : Institution PUBLIQUE ou ÉTATIQUE avec NOM PROPRE ou SIGLE identifiable :
                    - organisations internat. étatiques ("l'ONU", "l'OTAN", "l'UE", "l'UNESCO", "la BCE")
                    - institutions nationales nommées ("le Conseil constitutionnel", "la Cour suprême",
                      "le Sénat américain", "la Cour pénale internationale", "Interpol")
                    - corps publics avec nom officiel ("l'US Navy", "la Bundeswehr", "le GIGN")
                    - administrations nommées ("la CAF", "Pôle Emploi", "la SNCF" si publique)
                    RÈGLE : DOIT avoir un nom propre ou sigle figé reconnaissable.
                    NON si désigné de façon générique → hint_inst_role.

  hint_inst_role  : Institution PUBLIQUE désignée de façon GÉNÉRIQUE, sans nom propre :
                    - "le gouvernement", "la police", "l'armée", "le parlement", "le tribunal",
                      "les autorités", "la justice", "l'État", "la préfecture", "le ministère",
                      "la mairie", "les services secrets", "la diplomatie", "la gendarmerie",
                      "le Congrès" (générique), "la marine" (générique), "l'aviation"
                    RÈGLE : si on peut mettre "un/une" devant → hint_inst_role.
                    RÈGLE : nom propre/sigle figé → hint_inst_name.
                    NON si organisation privée → hint_org_name.

  REMOVE          : UNIQUEMENT bruit manifeste : fragment coupé, déterminant seul (le/la/les/un),
                    chiffre nu, contenu illisible.

Pour chaque span numéroté, réponds UNIQUEMENT avec un tableau JSON compact.
Format :
  - label correct : {"id":N,"verdict":"OK"}
  - à corriger    : {"id":N,"verdict":"CHANGE","label":"hint_xxx","raison":"courte raison"}
  - bruit pur     : {"id":N,"verdict":"REMOVE","raison":"courte raison"}

Ne génère rien d'autre que le tableau JSON."""


# ── 1. Collecte des spans uniques ──────────────────────────────────────────────
unique = defaultdict(list)
span_original = {}

for split in ['train', 'val', 'test']:
    p = Path(f'data/{split}_v6.4.jsonl')
    if not p.exists():
        print(f"⚠ Fichier absent : {p}")
        continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            text = d['text']
            for span in d.get('spans', []):
                if span['label'] not in TARGET:
                    continue
                s, e = span['start'], span['end']
                span_text = text[s:e]
                key = (span['label'], span_text.lower().strip())
                span_original[key] = span_text
                if len(unique[key]) < 2:
                    ctx = (text[max(0,s-60):s] + '[[' + span_text + ']]'
                           + text[e:min(len(text),e+60)])
                    unique[key].append(ctx.replace('\n', ' '))

items = [{'label': k[0], 'span': span_original[k], 'span_key': k,
          'contexts': v} for k, v in unique.items()]
print(f"{len(items)} spans uniques  |  {(len(items)+BATCH_SIZE-1)//BATCH_SIZE} batches")

# ── 2. Reprise ─────────────────────────────────────────────────────────────────
done = {}
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            done[(r['label'], r['span'].lower().strip())] = r
    print(f"Reprise : {len(done)} déjà traités")

todo = [it for it in items if it['span_key'] not in done]
print(f"Reste à traiter : {len(todo)}")

# ── 3. Appel Claude ────────────────────────────────────────────────────────────
def call_claude(user_msg: str) -> str:
    payload = {
        "model":       "claude-haiku-4-5-20251001",
        "max_tokens":  4096,
        "system":      SYSTEM_PROMPT,
        "messages":    [{"role": "user", "content": user_msg}],
        "temperature": 0.0,
    }
    for attempt in range(5):
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
        if r.status_code == 429:
            wait = 8 * (2 ** attempt)
            print(f"  429 → attente {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()['content'][0]['text'].strip()
    raise RuntimeError("5 tentatives épuisées (429)")

def parse_response(response: str, batch: list) -> list:
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if not match:
        m2 = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if m2:
            class _M:
                def group(self, n=0): return m2.group(1)
            match = _M()
    if not match:
        raise ValueError(f"Pas de tableau JSON dans: {response[:300]}")
    verdicts = json.loads(match.group())
    if isinstance(verdicts, dict):
        verdicts = [verdicts]
    verdicts_by_id = {v['id']: v for v in verdicts}
    results = []
    for i, item in enumerate(batch, start=1):
        v = verdicts_by_id.get(i, {'id': i, 'verdict': 'PARSE_MISSING'})
        verdict = v.get('verdict', 'PARSE_MISSING')
        results.append({
            'label':           item['label'],
            'span':            item['span'],
            'verdict':         verdict,
            'label_suggested': v.get('label') if verdict == 'CHANGE' else None,
            'raison':          v.get('raison'),
        })
    return results

# ── 4. Envoi par batches ───────────────────────────────────────────────────────
success = errors = 0
batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

with open(RESULTS_FILE, 'a', encoding='utf-8') as fout:
    for bi, batch in enumerate(batches):
        lines = []
        for i, item in enumerate(batch, start=1):
            ctx_str = ' | '.join(item['contexts'])
            lines.append(f'{i}. [{item["label"]}] "{item["span"]}" — {ctx_str}')
        user_msg = "Évalue ces spans NER :\n" + "\n".join(lines)

        print(f"[batch {bi+1}/{len(batches)}] {len(batch)} spans...", end=' ', flush=True)
        try:
            response = call_claude(user_msg)
            results  = parse_response(response, batch)
            changes  = sum(1 for r in results if r['verdict'] != 'OK')
            removes  = sum(1 for r in results if r['verdict'] == 'REMOVE')
            print(f"{changes} à changer ({removes} REMOVE)")
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            fout.flush()
            success += len(results)
        except Exception as e:
            print(f"ERREUR: {e}")
            for item in batch:
                r = {'label': item['label'], 'span': item['span'],
                     'verdict': 'ERROR', 'error': str(e)}
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            errors += len(batch)

        time.sleep(SLEEP_SEC)

print(f"\n✅ {success} traités, {errors} erreurs → {RESULTS_FILE}")

# ── 5. Résumé ──────────────────────────────────────────────────────────────────
from collections import Counter
all_res = [json.loads(l) for l in open(RESULTS_FILE, encoding='utf-8')]

verdict_counts = Counter(r['verdict'] for r in all_res)
print(f"\n=== RÉSUMÉ ===")
for v, n in sorted(verdict_counts.items(), key=lambda x: -x[1]):
    print(f"  {v:15s} : {n}")

change_from = defaultdict(Counter)
for r in all_res:
    if r['verdict'] == 'CHANGE' and r.get('label_suggested'):
        change_from[r['label']][r['label_suggested']] += 1

print(f"\n=== CHANGEMENTS SUGGÉRÉS ===")
for lbl, targets in sorted(change_from.items()):
    print(f"  {lbl}:")
    for tgt, n in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"    -> {tgt} : {n}")

print(f"\nTop 50 CHANGE :")
notable = [(r['label'], r['span'], r.get('label_suggested',''), r.get('raison','')[:75])
           for r in all_res if r['verdict'] == 'CHANGE']
notable.sort(key=lambda x: x[0])
for lbl, span, sugg, raison in notable[:50]:
    print(f"  [{lbl}] \"{span}\" -> {sugg} | {raison}")

