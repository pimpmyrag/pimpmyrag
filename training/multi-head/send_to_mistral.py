"""
Envoie le batch à Mistral Large et collecte les réponses.
Utilise l'API REST directe (requests).
"""
import json, time, requests
from pathlib import Path

API_KEY = "sX7fpLMqKqXbHx5bRtxTZSkjK0EWfdnF"
API_URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

batch_in = Path('data/suspects_mistral_filtered.jsonl')
results_out = Path('data/mistral_review_results.jsonl')

# Charge les requêtes
batch_requests = []
with open(batch_in, encoding='utf-8') as f:
    for line in f:
        batch_requests.append(json.loads(line))

print(f"Envoi de {len(batch_requests)} requêtes à Mistral Large...")
print(f"Coût estimé: ~{len(batch_requests) * 0.5:.1f}€ (58k input + ~10k output)")

import re

# Reprise: charge les span_keys déjà traités
done_keys = set()
if results_out.exists():
    with open(results_out, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            done_keys.add(r['span_key'])
    print(f"Reprise: {len(done_keys)} déjà traités, on continue...")

results = []
success = 0
errors = 0

for i, req in enumerate(batch_requests):
    span_key = req['span_key']

    if span_key in done_keys:
        print(f"[{i+1}/{len(batch_requests)}] SKIP {span_key[:50]}")
        continue

    print(f"[{i+1}/{len(batch_requests)}] {span_key[:60]}", flush=True)

    payload = {
        "model": "mistral-large-latest",
        "messages": req['messages']
    }

    # Retry avec backoff exponentiel sur 429
    for attempt in range(5):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80s
                print(f"  429 Rate limit, attente {wait}s (tentative {attempt+1}/5)...", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if attempt == 4:
                print(f"  ERREUR finale sur {span_key}: {e}")
                result = {'span_key': span_key, 'current_label': req['current_label'], 'error': str(e)}
                results.append(result)
                errors += 1
                # Sauvegarde immédiate
                with open(results_out, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                break
        except Exception as e:
            print(f"  ERREUR sur {span_key}: {e}")
            result = {'span_key': span_key, 'current_label': req['current_label'], 'error': str(e)}
            results.append(result)
            errors += 1
            with open(results_out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            break
    else:
        # Toutes les tentatives épuisées
        time.sleep(1.0)
        continue

    try:
        data = response.json()
        content = data['choices'][0]['message']['content'].strip()

        # Parse JSON de la réponse
        try:
            answer = json.loads(content)
        except:
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                answer = json.loads(match.group(1))
            else:
                answer = {'label': 'PARSE_ERROR', 'raison': content[:200]}

        result = {
            'span_key': span_key,
            'current_label': req['current_label'],
            'category': req['category'],
            'count': req['count'],
            'mistral_label': answer.get('label'),
            'mistral_raison': answer.get('raison'),
            'raw_response': content
        }
        results.append(result)
        success += 1

        # Sauvegarde immédiate (reprendre en cas d'interruption)
        with open(results_out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"  PARSE ERROR sur {span_key}: {e}")
        result = {'span_key': span_key, 'current_label': req['current_label'], 'error': str(e)}
        results.append(result)
        errors += 1
        with open(results_out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Rate limit: 1 req/s
    time.sleep(1.0)

print(f"\n\n✅ Terminé: {success} succès, {errors} erreurs")

print(f"Résultats écrits: {results_out}")

# Résumé des changements
print("\n" + "="*70)
print("RÉSUMÉ DES CORRECTIONS PROPOSÉES")
print("="*70)
changes = {'same': 0, 'changed': 0, 'error': 0}
for r in results:
    if 'error' in r:
        changes['error'] += 1
    elif r.get('mistral_label') == r.get('current_label'):
        changes['same'] += 1
    else:
        changes['changed'] += 1
        print(f"  {r['span_key']}")
        print(f"    AVANT: {r['current_label']}")
        print(f"    APRÈS: {r['mistral_label']}")
        print(f"    RAISON: {r.get('mistral_raison', 'N/A')[:80]}")
        print()

print(f"\nInchangés: {changes['same']}, Modifiés: {changes['changed']}, Erreurs: {changes['error']}")

