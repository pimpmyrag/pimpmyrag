import json, torch
from torch.utils.data import Dataset

LABELS = {
'hint_person_name':0,'hint_person_role':1,'hint_norp':2,'hint_group_role':3,'hint_org_name':4,
'hint_gpe':5,'hint_fac_name':6,'hint_loc_generic':7,'hint_weapon':8,'hint_vehicle':9,
'hint_substance':10,'hint_food':11,'hint_infra':12,'hint_tool':13,'hint_object_generic':14,
'hint_object_name':15,'hint_event_nominal':16,'hint_event_named':17,
'hint_time_date':18,'hint_time_clock':19,'hint_time_duration':20,'hint_quantity':21
}

# ── Coarse NER family (6 types) — déduit du label fin-grained ─────────────────
# Ordre = enum NerCoarseType dans le Kotlin : PER=0, LOC=1, ORG=2, TIME=3, EVENT=4, OBJECT=5
# Ce mapping est utilisé pour conditionner le SpanClassifier sur la famille coarse,
# SANS avoir à faire tourner le modèle coarse pendant le training (on le déduit du label).
NUM_COARSE   = 6
COARSE_NAMES = ['PER', 'LOC', 'ORG', 'TIME', 'EVENT', 'OBJECT']

LABEL_TO_COARSE: dict[str, int] = {
    # PER = 0
    'hint_person_name': 0, 'hint_person_role': 0,
    'hint_norp':        0, 'hint_group_role':  0,
    # LOC = 1
    'hint_gpe':         1, 'hint_fac_name':    1,
    'hint_loc_generic': 1, 'hint_infra':       1,
    # ORG = 2
    'hint_org_name':    2,
    # TIME = 3  — hint_quantity retiré (n'est pas une notion temporelle)
    'hint_time_date':     3, 'hint_time_clock':    3,
    'hint_time_duration': 3,
    # EVENT = 4
    'hint_event_nominal': 4, 'hint_event_named': 4,
    # OBJECT = 5  (défaut si label inconnu)
    'hint_weapon':         5, 'hint_vehicle':       5,
    'hint_substance':      5, 'hint_food':          5,
    'hint_tool':           5, 'hint_object_generic':5,
    'hint_object_name':    5, 'hint_quantity':      5,  # quantité = OBJECT, pas TIME
}
_COARSE_DEFAULT = 5  # OBJECT — fallback conservateur

# module-global tokenizer (will be set in SpanDataset.__init__)
TOKENIZER = None

class SpanDataset(Dataset):
    def __init__(self, path, tokenizer):
        # load lines safely
        records = []
        with open(path, 'r', encoding='utf-8') as fh:
            for l in fh:
                l = l.strip()
                if not l:
                    continue
                try:
                    records.append(json.loads(l))
                except Exception:
                    # skip malformed lines
                    continue
        # normalize records to have 'text' and 'spans'
        self.data = [self._normalize(r) for r in records]
        self.tokenizer = tokenizer
        # expose tokenizer globally for collate_fn compatibility
        global TOKENIZER
        TOKENIZER = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]

    def _normalize(self, obj):
        # Find text in several possible locations
        text = None
        # direct keys
        for k in ('text', 'phrase', 'sentence'):
            if k in obj and isinstance(obj[k], str):
                text = obj[k]
                break
        # nested under data
        if text is None and 'data' in obj and isinstance(obj['data'], dict):
            if 'text' in obj['data'] and isinstance(obj['data']['text'], str):
                text = obj['data']['text']
        # nested under body/message
        if text is None and 'body' in obj and isinstance(obj['body'], dict):
            # try common shapes
            b = obj['body']
            if 'text' in b and isinstance(b['text'], str):
                text = b['text']
            elif 'messages' in b and isinstance(b['messages'], list) and len(b['messages'])>0:
                # take last user content if present
                for m in reversed(b['messages']):
                    if isinstance(m, dict) and 'content' in m and isinstance(m['content'], str):
                        # try to extract phrase after 'Phrase :' if present
                        content = m['content']
                        if 'Phrase :' in content:
                            cand = content.split('Phrase :',1)[1].strip()
                            if cand:
                                text = cand
                                break
                        # fallback: take whole content
                        text = content
                        break
        if text is None:
            # fallback: try 'id' or empty
            text = obj.get('text', '') or obj.get('phrase', '') or ''
        # find spans in several locations
        spans = None
        if 'spans' in obj and isinstance(obj['spans'], list):
            spans = obj['spans']
        elif 'json' in obj and isinstance(obj['json'], dict) and 'spans' in obj['json']:
            spans = obj['json']['spans']
        elif 'data' in obj and isinstance(obj['data'], dict) and 'spans' in obj['data']:
            spans = obj['data']['spans']
        else:
            spans = []
        # ensure spans are list of dicts with expected keys
        clean_spans = []
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            # allow label under 'label' or 'type'
            label = sp.get('label') or sp.get('type') or None
            start = sp.get('start')
            end = sp.get('end')
            text_span = sp.get('text') or sp.get('span') or None
            clean_spans.append({'label': label, 'start': start, 'end': end, 'text': text_span})
        return {'text': text, 'spans': clean_spans}


def _map_char_span_to_token_span(offsets, char_start, char_end):
    """Map character offsets to token index span using offsets list for a single example.
    offsets: list of (start,end) tuples per token (includes special tokens, may have (0,0)).
    Returns (token_start, token_end) as token indices where end is exclusive, or None if not mappable.
    """
    token_start = None
    token_end = None
    # find token_start: first token with end > char_start and start != end
    for i, (ts, te) in enumerate(offsets):
        if ts == te == 0:
            continue
        if te > char_start and ts <= char_start:
            token_start = i
            break
    # find token_end: first token with end >= char_end
    for j, (ts, te) in enumerate(offsets):
        if ts == te == 0:
            continue
        if te >= char_end and ts < char_end:
            token_end = j + 1
            break
    # if not found, attempt fallback heuristics
    if token_start is None or token_end is None:
        return None
    if token_start >= token_end:
        return None
    return (token_start, token_end)


def collate_fn(batch):
    # expects batch of normalized items with 'text' and 'spans'
    global TOKENIZER
    if TOKENIZER is None:
        raise RuntimeError('TOKENIZER not set in dataset module. Pass tokenizer to SpanDataset to set it.')
    texts = []
    raw_spans = []
    for item in batch:
        t = None
        if isinstance(item, dict):
            t = item.get('text') or item.get('phrase')
            if not t and 'data' in item and isinstance(item['data'], dict):
                t = item['data'].get('text')
            if not t and 'json' in item and isinstance(item['json'], dict):
                t = item['json'].get('text')
            if not t:
                t = ''
            s = item.get('spans') or []
            if not isinstance(s, list):
                s = []
        else:
            t = str(item)
            s = []
        texts.append(t)
        raw_spans.append(s)

    # Tokenize with offsets (fast tokenizer required)
    enc = TOKENIZER(texts, padding=True, truncation=True, max_length=256, return_attention_mask=True, return_offsets_mapping=True)

    # convert input_ids and attention_mask to tensors
    input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)

    # offsets: list of list of (start,end)
    offsets_batch = enc['offset_mapping']

    spans_tokenized = []  # per-sample list of token spans
    flat_labels = []
    for i, sp_list in enumerate(raw_spans):
        offs = offsets_batch[i]
        sample_token_spans = []
        if not isinstance(sp_list, list):
            sp_list = []
        for sp in sp_list:
            if not isinstance(sp, dict):
                continue
            label_name = sp.get('label') or sp.get('type')
            if label_name not in LABELS:
                continue
            start = sp.get('start')
            end = sp.get('end')
            if start is None or end is None:
                continue
            try:
                start = int(start)
                end = int(end)
            except Exception:
                continue
            mapped = _map_char_span_to_token_span(offs, start, end)
            if mapped is None:
                # skip spans that cannot be mapped to tokens
                continue
            token_start, token_end = mapped
            coarse_id = LABEL_TO_COARSE.get(label_name, _COARSE_DEFAULT)
            sample_token_spans.append({'start': token_start, 'end': token_end, 'coarse_id': coarse_id})
            flat_labels.append(LABELS[label_name])
        spans_tokenized.append(sample_token_spans)

    if len(flat_labels) == 0:
        labels_tensor = torch.empty((0,), dtype=torch.long)
    else:
        labels_tensor = torch.tensor(flat_labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'spans': spans_tokenized,
        'labels': labels_tensor,
        'loss_fn': torch.nn.CrossEntropyLoss()
    }
