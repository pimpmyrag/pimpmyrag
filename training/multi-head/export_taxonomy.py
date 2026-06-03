#!/usr/bin/env python3
"""Export taxonomy docs from labels.py.

Source of truth: training/multi-head/labels.py
Outputs:
- docs/taxonomy.json
- docs/taxonomy.schema.json
- docs/TAXONOMY.md

Run from repo root or from training/multi-head:
    python3 training/multi-head/export_taxonomy.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import labels


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TAXONOMY_JSON = DOCS / "taxonomy.json"
TAXONOMY_SCHEMA = DOCS / "taxonomy.schema.json"
TAXONOMY_MD = DOCS / "TAXONOMY.md"


def label_items(values: list[str]) -> list[dict]:
    return [{"id": i, "label": value} for i, value in enumerate(values)]


def mapping_to_label_names(mapping: dict[int, list[int]], target_labels: list[str]) -> dict[str, list[str]]:
    return {
        str(k): [target_labels[i] for i in v]
        for k, v in mapping.items()
    }


def build_taxonomy() -> dict:
    coarse_to_fine = {
        labels.COARSE_LABELS[coarse_id]: [labels.FINE_LABELS[fine_id] for fine_id in fine_ids]
        for coarse_id, fine_ids in labels.COARSE_TO_FINE.items()
    }
    role_fine_to_coarse = {
        labels.ROLE_LABELS[fine_id]: labels.ROLE_COARSE_LABELS[coarse_id]
        for fine_id, coarse_id in labels.ROLE_FINE_TO_COARSE_ID.items()
    }
    role_to_oblique = {
        labels.ROLE_LABELS[role_id]: labels.ROLE_OBLIQUE_LABELS[obl_id]
        for role_id, obl_id in labels.ROLE_TO_OBLIQUE_ID.items()
    }
    verb_family_to_fine = {
        labels.VERB_FAMILY_LABELS[family_id]: [labels.VERB_FAMILY_FINE_LABELS[fine_id] for fine_id in fine_ids]
        for family_id, fine_ids in labels.VERB_FAMILY_TO_FINE.items()
    }

    return {
        "$schema": "./taxonomy.schema.json",
        "source": "training/multi-head/labels.py",
        "generated_on": date.today().isoformat(),
        "ner": {
            "coarse": {
                "count": len(labels.COARSE_LABELS),
                "none_id": labels.COARSE_NONE_ID,
                "labels": label_items(labels.COARSE_LABELS),
            },
            "fine": {
                "count": len(labels.FINE_LABELS),
                "none_id": labels.FINE_NONE_ID,
                "labels": label_items(labels.FINE_LABELS),
                "groups": {
                    "concrete": [labels.FINE_LABELS[i] for i in sorted(labels.FINE_CONCRETE_IDS)],
                    "abstract": [labels.FINE_LABELS[i] for i in sorted(labels.FINE_ABSTRACT_IDS)],
                },
            },
            "coarse_to_fine": coarse_to_fine,
        },
        "syntax": {
            "span_labels": {
                "count": len(labels.SYN_LABELS),
                "none_id": labels.SYN_NONE_ID,
                "labels": label_items(labels.SYN_LABELS),
            },
            "role": {
                "count": len(labels.ROLE_LABELS),
                "none_id": labels.ROLE_NONE_ID,
                "labels": label_items(labels.ROLE_LABELS),
            },
            "role_coarse": {
                "count": len(labels.ROLE_COARSE_LABELS),
                "none_id": labels.ROLE_COARSE_NONE_ID,
                "other_id": labels.ROLE_COARSE_OTHER_ID,
                "labels": label_items(labels.ROLE_COARSE_LABELS),
            },
            "role_oblique": {
                "count": len(labels.ROLE_OBLIQUE_LABELS),
                "none_id": labels.ROLE_OBLIQUE_NONE_ID,
                "labels": label_items(labels.ROLE_OBLIQUE_LABELS),
            },
            "role_fine_to_coarse": role_fine_to_coarse,
            "role_to_oblique": role_to_oblique,
        },
        "verb": {
            "family": {
                "count": len(labels.VERB_FAMILY_LABELS),
                "none_id": labels.VERB_FAMILY_NONE_ID,
                "labels": label_items(labels.VERB_FAMILY_LABELS),
            },
            "family_fine": {
                "count": len(labels.VERB_FAMILY_FINE_LABELS),
                "none_id": labels.VERB_FAMILY_FINE_NONE_ID,
                "labels": label_items(labels.VERB_FAMILY_FINE_LABELS),
            },
            "family_to_fine": verb_family_to_fine,
            "polarity": {
                "count": len(labels.VERB_POLARITY_LABELS),
                "none_id": labels.VERB_POLARITY_NONE_ID,
                "labels": label_items(labels.VERB_POLARITY_LABELS),
            },
            "aspect": {
                "count": len(labels.VERB_ASPECT_LABELS),
                "none_id": labels.VERB_ASPECT_NONE_ID,
                "labels": label_items(labels.VERB_ASPECT_LABELS),
            },
            "source": {
                "count": len(labels.VERB_SOURCE_LABELS),
                "none_id": labels.VERB_SOURCE_NONE_ID,
                "labels": label_items(labels.VERB_SOURCE_LABELS),
            },
        },
        "morphology": {
            "voice": {
                "count": len(labels.VOICE_LABELS),
                "none_id": labels.VOICE_NONE_ID,
                "labels": label_items(labels.VOICE_LABELS),
            },
            "certainty": {
                "count": len(labels.CERTAINTY_LABELS),
                "none_id": labels.CERTAINTY_NONE_ID,
                "labels": label_items(labels.CERTAINTY_LABELS),
            },
            "gender": {
                "count": len(labels.GENDER_LABELS),
                "none_id": labels.GENDER_NONE_ID,
                "labels": label_items(labels.GENDER_LABELS),
            },
            "number": {
                "count": len(labels.NUMBER_LABELS),
                "none_id": labels.NUMBER_NONE_ID,
                "labels": label_items(labels.NUMBER_LABELS),
            },
            "person": {
                "count": len(labels.PERSON_LABELS),
                "none_id": labels.PERSON_NONE_ID,
                "labels": label_items(labels.PERSON_LABELS),
            },
        },
        "notes": [
            "Sentinel none_id values are outside the active label range unless explicitly part of labels (ROLE_LABELS contains NONE at id 6).",
            "role is the current primary SVO role head (12 labels). role_coarse and role_oblique are auxiliary/cascade heads and may be trained with lambda=0 in selected configs.",
            "COARSE_LABELS intentionally includes NONE at id 9 for model compatibility.",
        ],
    }


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/pimpmyrag/pimpmyrag/docs/taxonomy.schema.json",
    "title": "PimpMyRAG taxonomy export",
    "type": "object",
    "required": ["source", "generated_on", "ner", "syntax", "verb", "morphology"],
    "properties": {
        "$schema": {"type": "string"},
        "source": {"type": "string"},
        "generated_on": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "ner": {"$ref": "#/$defs/ner"},
        "syntax": {"$ref": "#/$defs/syntax"},
        "verb": {"$ref": "#/$defs/verb"},
        "morphology": {"$ref": "#/$defs/morphology"},
    },
    "$defs": {
        "labelItem": {
            "type": "object",
            "required": ["id", "label"],
            "properties": {"id": {"type": "integer", "minimum": 0}, "label": {"type": "string"}},
            "additionalProperties": False,
        },
        "labelSet": {
            "type": "object",
            "required": ["count", "none_id", "labels"],
            "properties": {
                "count": {"type": "integer", "minimum": 0},
                "none_id": {"type": "integer", "minimum": 0},
                "other_id": {"type": "integer", "minimum": 0},
                "labels": {"type": "array", "items": {"$ref": "#/$defs/labelItem"}},
                "groups": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
            },
            "additionalProperties": False,
        },
        "mapping": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
        "ner": {
            "type": "object",
            "required": ["coarse", "fine", "coarse_to_fine"],
            "properties": {
                "coarse": {"$ref": "#/$defs/labelSet"},
                "fine": {"$ref": "#/$defs/labelSet"},
                "coarse_to_fine": {"$ref": "#/$defs/mapping"},
            },
            "additionalProperties": False,
        },
        "syntax": {
            "type": "object",
            "required": ["span_labels", "role", "role_coarse", "role_oblique", "role_fine_to_coarse", "role_to_oblique"],
            "properties": {
                "span_labels": {"$ref": "#/$defs/labelSet"},
                "role": {"$ref": "#/$defs/labelSet"},
                "role_coarse": {"$ref": "#/$defs/labelSet"},
                "role_oblique": {"$ref": "#/$defs/labelSet"},
                "role_fine_to_coarse": {"type": "object", "additionalProperties": {"type": "string"}},
                "role_to_oblique": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "verb": {
            "type": "object",
            "required": ["family", "family_fine", "family_to_fine", "polarity", "aspect", "source"],
            "properties": {
                "family": {"$ref": "#/$defs/labelSet"},
                "family_fine": {"$ref": "#/$defs/labelSet"},
                "family_to_fine": {"$ref": "#/$defs/mapping"},
                "polarity": {"$ref": "#/$defs/labelSet"},
                "aspect": {"$ref": "#/$defs/labelSet"},
                "source": {"$ref": "#/$defs/labelSet"},
            },
            "additionalProperties": False,
        },
        "morphology": {
            "type": "object",
            "required": ["voice", "certainty", "gender", "number", "person"],
            "properties": {
                "voice": {"$ref": "#/$defs/labelSet"},
                "certainty": {"$ref": "#/$defs/labelSet"},
                "gender": {"$ref": "#/$defs/labelSet"},
                "number": {"$ref": "#/$defs/labelSet"},
                "person": {"$ref": "#/$defs/labelSet"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def labels_md(label_set: dict) -> str:
    return ", ".join(f"`{item['id']}:{item['label']}`" for item in label_set["labels"])


def mapping_md(mapping: dict[str, list[str]]) -> str:
    lines = []
    for key, values in mapping.items():
        lines.append(f"- `{key}` → " + ", ".join(f"`{v}`" for v in values))
    return "\n".join(lines)


def build_markdown(taxonomy: dict) -> str:
    ner = taxonomy["ner"]
    syntax = taxonomy["syntax"]
    verb = taxonomy["verb"]
    morph = taxonomy["morphology"]
    return f"""# PimpMyRAG taxonomy

> Generated from `training/multi-head/labels.py` on {taxonomy['generated_on']}.
> Do not edit label lists manually here; run `python3 training/multi-head/export_taxonomy.py` after changing `labels.py`.

## Source of truth

- Python taxonomy: `training/multi-head/labels.py`
- Machine-readable export: `docs/taxonomy.json`
- JSON Schema: `docs/taxonomy.schema.json`

## Summary

| Family | Active labels | Sentinel / note |
|---|---:|---|
| NER coarse | {ner['coarse']['count']} | `NONE` is an active model label at id `{ner['coarse']['none_id']}` |
| NER fine | {ner['fine']['count']} | sentinel `FINE_NONE_ID={ner['fine']['none_id']}` outside active range |
| Syntax spans | {syntax['span_labels']['count']} | sentinel `{syntax['span_labels']['none_id']}` |
| Role 12 labels | {syntax['role']['count']} | `NONE` is label id `{syntax['role']['none_id']}`; primary role head |
| Role coarse | {syntax['role_coarse']['count']} | auxiliary/cascade head; sentinel `{syntax['role_coarse']['none_id']}` |
| Role oblique | {syntax['role_oblique']['count']} | auxiliary/cascade head; sentinel `{syntax['role_oblique']['none_id']}` |
| Verb family | {verb['family']['count']} | sentinel `{verb['family']['none_id']}` |
| Verb family fine | {verb['family_fine']['count']} | sentinel `{verb['family_fine']['none_id']}` |
| Verb polarity/aspect/source | {verb['polarity']['count']} / {verb['aspect']['count']} / {verb['source']['count']} | verb-trigger only |
| Voice/certainty | {morph['voice']['count']} / {morph['certainty']['count']} | verb-trigger only |
| Gender/number/person | {morph['gender']['count']} / {morph['number']['count']} / {morph['person']['count']} | supervised where annotated |

## NER

### Coarse labels

{labels_md(ner['coarse'])}

### Fine labels

{labels_md(ner['fine'])}

### Coarse → fine

{mapping_md(ner['coarse_to_fine'])}

## Syntax / SVO

### Syntax span labels

{labels_md(syntax['span_labels'])}

### Role head — 12 labels, primary

{labels_md(syntax['role'])}

The 12-label role head is currently the primary SVO role classifier. It already encodes `SUBJECT`, `OBJECT`, `APPOS`, generic `OBLIQUE`, and the fine `OBLIQUE_*` roles.

### Auxiliary role coarse

{labels_md(syntax['role_coarse'])}

`role_coarse` is kept for diagnostics/cascade experiments. Some training configs set its lambda to `0.0` when the 12-label `role` head is sufficient.

### Auxiliary oblique fine

{labels_md(syntax['role_oblique'])}

`role_oblique` is an auxiliary/cascade head conditioned on oblique spans. Some training configs set its lambda to `0.0` to avoid redundant loss budget.

## Verb taxonomy

### Verb family

{labels_md(verb['family'])}

### Verb family fine

{labels_md(verb['family_fine'])}

### Verb family → fine

{mapping_md(verb['family_to_fine'])}

### Verb polarity / aspect / source

- Polarity: {labels_md(verb['polarity'])}
- Aspect: {labels_md(verb['aspect'])}
- Source: {labels_md(verb['source'])}

## Morphology / modality

- Voice: {labels_md(morph['voice'])}
- Certainty: {labels_md(morph['certainty'])}
- Gender: {labels_md(morph['gender'])}
- Number: {labels_md(morph['number'])}
- Person: {labels_md(morph['person'])}

## Maintenance checklist

When changing taxonomy:

1. Update `training/multi-head/labels.py` only.
2. Run:

```zsh
cd pimpmyrag
source training/multi-head/venv/bin/activate
python3 training/multi-head/export_taxonomy.py
```

3. Commit `labels.py`, `docs/taxonomy.json`, `docs/taxonomy.schema.json`, and `docs/TAXONOMY.md` together.
"""


def validate_taxonomy(taxonomy: dict) -> None:
    assert taxonomy["ner"]["coarse"]["count"] == len(taxonomy["ner"]["coarse"]["labels"])
    assert taxonomy["ner"]["fine"]["count"] == len(taxonomy["ner"]["fine"]["labels"])
    assert taxonomy["syntax"]["role"]["count"] == len(taxonomy["syntax"]["role"]["labels"])
    assert taxonomy["verb"]["family"]["count"] == len(taxonomy["verb"]["family"]["labels"])
    assert taxonomy["verb"]["family_fine"]["count"] == len(taxonomy["verb"]["family_fine"]["labels"])


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    taxonomy = build_taxonomy()
    validate_taxonomy(taxonomy)
    TAXONOMY_SCHEMA.write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TAXONOMY_JSON.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TAXONOMY_MD.write_text(build_markdown(taxonomy), encoding="utf-8")
    print(f"Wrote {TAXONOMY_JSON.relative_to(ROOT)}")
    print(f"Wrote {TAXONOMY_SCHEMA.relative_to(ROOT)}")
    print(f"Wrote {TAXONOMY_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

