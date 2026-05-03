"""
Template générique de batch review NER via Claude Batch API.
50% moins cher + parallèle vs l'API synchrone.

Usage :
  python3 scripts/batch_review_template.py \
    --input  data/inst_labels_review_input.jsonl \
    --output data/inst_labels_review.jsonl \
    --target hint_inst_name,hint_inst_role,hint_org_name \
    --system-prompt-file scripts/prompts/review_inst.txt  # ou inline via SYSTEM_PROMPT

Fichier input attendu : JSONL avec champs {label, span, contexts: [str, str]}
Fichier output        : JSONL avec champs {label, span, verdict, label_suggested, raison}
"""
import argparse, json, os, re, time
from pathlib import Path
from collections import Counter, defaultdict

import httpx

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_secrets(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

def build_requests_jsonl(items: list[dict], batch_size: int,
                         system_prompt: str, model: str,
                         out_path: Path):
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    with open(out_path, 'w', encoding='utf-8') as f:
        for bi, batch in enumerate(batches):
            lines = []
            for i, item in enumerate(batch, start=1):
                ctx_str = ' | '.join(item.get('contexts', []))
                lines.append(f'{i}. [{item["label"]}] "{item["span"]}" — {ctx_str}')
            user_msg = "Évalue ces spans NER :\n" + "\n".join(lines)
            req = {
                "custom_id": f"batch_{bi}",
                "params": {
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "system": [{"type": "text", "text": system_prompt,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": user_msg}],
                }
            }
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f"📦 {len(batches)} requêtes → {out_path}")
    return batches

def submit(api_key: str, requests_jsonl: Path) -> str:
    url = "https://api.anthropic.com/v1/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "message-batches-2024-09-24,prompt-caching-2024-07-31",
    }
    reqs = [json.loads(l) for l in requests_jsonl.read_text().splitlines() if l.strip()]
    print(f"📤 Envoi de {len(reqs)} requêtes...")
    with httpx.Client(timeout=120) as c:
        r = c.post(url, headers=headers, json={"requests": reqs})
        r.raise_for_status()
    batch_id = r.json()["id"]
    id_file = requests_jsonl.with_suffix('.batch_id')
    id_file.write_text(batch_id)
    print(f"✅ Batch : {batch_id}  (ID sauvegardé dans {id_file})")
    return batch_id

def poll(api_key: str, batch_id: str, interval: int = 20) -> None:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "anthropic-beta": "message-batches-2024-09-24"}
    t0 = time.time()
    while True:
        with httpx.Client(timeout=60) as c:
            data = c.get(url, headers=headers).json()
        status = data.get("processing_status")
        counts = data.get("request_counts", {})
        print(f"  ⏳ [{time.time()-t0:.0f}s] {status} | "
              f"✅{counts.get('succeeded',0)} ❌{counts.get('errored',0)} "
              f"🔄{counts.get('processing',0)}")
        if status == "ended":
            print(f"🎉 Terminé en {time.time()-t0:.0f}s")
            return
        time.sleep(interval)

def fetch(api_key: str, batch_id: str) -> list[dict]:
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "anthropic-beta": "message-batches-2024-09-24"}
    results = []
    with httpx.Client(timeout=120) as c:
        with c.stream("GET", url, headers=headers) as r:
            r.raise_for_status()
            buf = ""
            for chunk in r.iter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    print(f"📥 {len(results)} résultats récupérés")
    return results

def parse_verdicts(response_text: str, batch: list[dict]) -> list[dict]:
    text = response_text.strip()
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        m2 = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if m2:
            text = m2.group(1)
            m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        raise ValueError(f"Pas de JSON dans: {text[:200]}")
    verdicts = json.loads(m.group())
    if isinstance(verdicts, dict):
        verdicts = [verdicts]
    by_id = {v['id']: v for v in verdicts}
    results = []
    for i, item in enumerate(batch, start=1):
        v = by_id.get(i, {'id': i, 'verdict': 'PARSE_MISSING'})
        verdict = v.get('verdict', 'PARSE_MISSING')
        results.append({
            'label':           item['label'],
            'span':            item['span'],
            'verdict':         verdict,
            'label_suggested': v.get('label') if verdict == 'CHANGE' else None,
            'raison':          v.get('raison'),
        })
    return results

def process_results(raw_results: list[dict], batches: list[list[dict]],
                    out_path: Path):
    batch_map = {f"batch_{i}": b for i, b in enumerate(batches)}
    all_records = []
    errors = 0
    for res in raw_results:
        cid    = res.get("custom_id", "")
        rtype  = res.get("result", {}).get("type", "")
        batch  = batch_map.get(cid, [])
        if rtype == "succeeded":
            text = "".join(b["text"] for b in
                           res["result"]["message"].get("content", [])
                           if b.get("type") == "text")
            try:
                records = parse_verdicts(text, batch)
            except Exception as e:
                print(f"  ⚠ parse erreur {cid}: {e}")
                records = [{'label': it['label'], 'span': it['span'],
                            'verdict': 'PARSE_ERROR', 'error': str(e)}
                           for it in batch]
        else:
            errors += 1
            msg = res.get("result", {}).get("error", {}).get("message", "")
            print(f"  ❌ {cid}: {msg}")
            records = [{'label': it['label'], 'span': it['span'],
                        'verdict': 'ERROR', 'error': msg} for it in batch]
        all_records.extend(records)

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    vc = Counter(r['verdict'] for r in all_records)
    print(f"\n✅ {out_path}  ({len(all_records)} spans, {errors} batches en erreur)")
    print("=== RÉSUMÉ ===")
    for v, n in sorted(vc.items(), key=lambda x: -x[1]):
        print(f"  {v:15s} : {n}")

    change_from = defaultdict(Counter)
    for r in all_records:
        if r['verdict'] == 'CHANGE' and r.get('label_suggested'):
            change_from[r['label']][r['label_suggested']] += 1
    if change_from:
        print("=== CHANGEMENTS ===")
        for src, tgts in sorted(change_from.items()):
            for tgt, n in sorted(tgts.items(), key=lambda x: -x[1]):
                print(f"  {src} → {tgt} : {n}")
    return all_records

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True, help="JSONL {label,span,contexts}")
    ap.add_argument("--output",  required=True, help="JSONL résultats")
    ap.add_argument("--system-prompt", required=True, help="Fichier texte ou string inline")
    ap.add_argument("--model",   default="claude-haiku-4-5-20251001")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--requests-file", default=None)
    ap.add_argument("--batch-id", default=None, help="Reprendre un batch existant")
    ap.add_argument("--secrets", default=".secrets.env")
    args = ap.parse_args()

    load_secrets(Path(args.secrets))
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Charger le system prompt
    sp_path = Path(args.system_prompt)
    system_prompt = sp_path.read_text(encoding='utf-8') if sp_path.exists() else args.system_prompt

    # Charger le fichier input
    items = [json.loads(l) for l in Path(args.input).read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"📝 {len(items)} spans à évaluer")

    req_file = Path(args.requests_file or args.output.replace('.jsonl', '_requests.jsonl'))

    if not args.batch_id:
        saved_id_file = req_file.with_suffix('.batch_id')
        if saved_id_file.exists():
            args.batch_id = saved_id_file.read_text().strip()
            print(f"🔄 Reprise batch {args.batch_id}")

    batches = [items[i:i+args.batch_size] for i in range(0, len(items), args.batch_size)]

    if not args.batch_id:
        build_requests_jsonl(items, args.batch_size, system_prompt, args.model, req_file)
        args.batch_id = submit(api_key, req_file)

    poll(api_key, args.batch_id)
    raw = fetch(api_key, args.batch_id)
    process_results(raw, batches, Path(args.output))

if __name__ == "__main__":
    main()

