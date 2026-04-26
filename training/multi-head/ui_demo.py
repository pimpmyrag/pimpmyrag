"""
ui_demo.py — Interface visuelle pour le modèle NER + SVO multitête.

Lancer :
    python ui_demo.py

Fonctionnalités :
  • Coller du texte → analyse NER + SVO en batch
  • Texte surligné : couleurs par catégorie coarse (NER) et rôle SVO
  • Clic sur un span → panneau détail complet (scores, morpho, type fin…)
  • Mode batch : coller plusieurs phrases (une par ligne)
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from test_model_sentences_v3 import load_model_and_tokenizer, predict_texts_batch, pick_device, post_process_dynamic, dedupe_overlaps
from labels import FINE_LABELS, COARSE_LABELS, COARSE_TO_FINE, FINE2ID

# ──────────────────────────────────────────────────────────
#  Config modèle
# ──────────────────────────────────────────────────────────

import os
_REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    str(Path(__file__).resolve().parents[2])
)
CHECKPOINT = os.environ.get(
    "NER_CHECKPOINT",
    os.path.join(_REPO_ROOT, "models", "deberta", "fine-tuning-22042026", "checkpoint_best_multitask.pt")
)
MODEL_NAME = "microsoft/deberta-v3-base"
TOKENIZER_PATH = None  # None = utilise MODEL_NAME
FORCE_DEVICE = "cpu"   # MPS buggé sur ce Mac — forcer CPU

# ──────────────────────────────────────────────────────────
#  Palette couleurs
# ──────────────────────────────────────────────────────────

# NER coarse
COARSE_COLORS = {
    "PER":      ("#dbeafe", "#1d4ed8"),   # bleu
    "LOC":      ("#d1fae5", "#065f46"),   # vert
    "ORG":      ("#ede9fe", "#5b21b6"),   # violet
    "TIME":     ("#ffedd5", "#9a3412"),   # orange
    "EVENT":    ("#fee2e2", "#991b1b"),   # rouge
    "VALUE":    ("#ccfbf1", "#0f766e"),   # teal
    "OBJECT":   ("#fef3c7", "#92400e"),   # jaune/brun
    "ABSTRACT": ("#f1f5f9", "#334155"),   # gris ardoise
    "NONE":     ("#f3f4f6", "#6b7280"),
}

# SVO rôles
SVO_COLORS = {
    "svo_verb":    ("#e0f2fe", "#0369a1"),   # bleu ciel
    "svo_subject": ("#dcfce7", "#15803d"),   # vert clair
    "svo_object":  ("#fce7f3", "#9d174d"),   # rose
    "svo_iobj":    ("#fff7ed", "#c2410c"),   # pêche
    "pron_subj":   ("#f0fdf4", "#166534"),   # vert pâle
    "pron_obj":    ("#fdf2f8", "#7e22ce"),   # mauve pâle
}

SVO_EMOJI = {
    "svo_verb":    "🔵", "svo_subject": "🟢", "svo_object": "🔴",
    "svo_iobj":    "🟠", "pron_subj":   "🟢", "pron_obj":   "🔴",
}

# ──────────────────────────────────────────────────────────
#  Splitter de phrases français
# ──────────────────────────────────────────────────────────

# Abréviations courantes qui ne terminent PAS une phrase
_ABBREVS = {
    "M", "Mme", "Mmes", "MM", "Dr", "Pr", "Prof", "art", "vol", "no", "n°",
    "p", "pp", "cf", "vs", "env", "fig", "éd", "janv", "févr", "mars", "avr",
    "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc", "St", "Ste",
    "av", "bd", "exc", "incl", "réf", "tél", "hab", "km", "cm", "mm",
}
_ABBREV_PAT = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREVS, key=len, reverse=True)) + r")\.$",
    re.IGNORECASE,
)

# Marqueur temporaire pour les fins de phrase réelles
_SENT_END = "\x00"

def split_sentences(text: str) -> list[str]:
    """
    Découpe un texte français en phrases.
    Gère : . ! ? … ; suivi d'espace + majuscule, guillemets, parenthèses.
    Préserve les abréviations et les nombres décimaux.
    """
    if not text.strip():
        return []

    # 1. Protéger les abréviations connues (M. → M\x01)
    protected = text
    for abbrev in sorted(_ABBREVS, key=len, reverse=True):
        protected = re.sub(
            r"\b" + re.escape(abbrev) + r"\.",
            abbrev + "\x01",
            protected,
            flags=re.IGNORECASE,
        )

    # 2. Protéger les nombres décimaux (3.14 → 3\x012)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x01" + m.group(2), protected)

    # 3. Protéger les ellipses
    protected = protected.replace("...", "\x02")

    # 4. Marquer les fins de phrase réelles
    #    Après . ! ? suivi de ) ou » ou espace+majuscule ou fin de chaîne
    protected = re.sub(
        r'([.!?])(["\u00bb\u2019\)]?)\s+(?=[A-ZÀÂÆÇÉÈÊËÎÏÔÙÛÜŒ\u00c0-\u00ff])',
        lambda m: m.group(1) + m.group(2) + _SENT_END,
        protected,
    )
    # Fin de chaîne après ponctuation
    protected = re.sub(r'([.!?])(["\u00bb\u2019\)]?)\s*$', lambda m: m.group(1) + m.group(2) + _SENT_END, protected)

    # 5. Découper sur le marqueur
    parts = protected.split(_SENT_END)

    # 6. Restaurer les protections
    sentences = []
    for p in parts:
        s = p.replace("\x01", ".").replace("\x02", "...").strip()
        if s:
            sentences.append(s)

    return sentences if sentences else [text.strip()]

_model = None
_tokenizer = None
_device = None

# Phrases warmup — courtes et variées pour forcer la compilation des kernels MPS
_WARMUP_PHRASES = [
    "Emmanuel Macron s'est rendu à Berlin pour rencontrer Olaf Scholz.",
    "La Banque centrale européenne a relevé ses taux d'intérêt de 25 points de base.",
    "Apple a annoncé le lancement de l'iPhone 17 le 15 septembre 2025 à Cupertino.",
    "Le tremblement de terre de magnitude 6,8 a touché la côte nord du Maroc.",
    "Tesla a livré 500 000 véhicules électriques au troisième trimestre.",
]

def get_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = FORCE_DEVICE  # pick_device() écarté — MPS instable sur ce Mac
        print(f"✅ device = {_device}")
        _model, _tokenizer = load_model_and_tokenizer(
            model_name=MODEL_NAME,
            checkpoint_path=CHECKPOINT,
            tokenizer_path=TOKENIZER_PATH,
            device=_device,
        )
        print("✅ Modèle chargé — warmup CPU en cours…")
        t0 = time.perf_counter()
        for _ in range(2):
            predict_texts_batch(
                model=_model, tokenizer=_tokenizer, texts=_WARMUP_PHRASES,
                device=_device, max_length=128, max_span_len=12,
                tau_boundary=0.5, tau_none=0.99, tau_coarse=0.7, tau_fine=0.7,
                topk_coarse=2, min_char_len=2, enforce_word_boundaries=True,
                tau_svo_boundary=0.5,
            )
        print(f"✅ Warmup terminé en {time.perf_counter()-t0:.1f}s")
    return _model, _tokenizer, _device


# ──────────────────────────────────────────────────────────
#  Helpers HTML
# ──────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _confidence_badge(data: dict) -> str:
    """Pastille colorée selon le score de confiance du span."""
    # NER → score global ; SVO → svo_prob (ou svo_boundary_prob si absent)
    score = data.get("score") or data.get("svo_prob") or data.get("svo_boundary_prob")
    if score is None:
        return ""
    if score >= 0.85:
        color, title = "#16a34a", f"Confiance haute ({score:.2f})"   # vert
    elif score >= 0.60:
        color, title = "#d97706", f"Confiance moyenne ({score:.2f})" # orange
    else:
        color, title = "#dc2626", f"Confiance basse ({score:.2f})"   # rouge
    return (
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{color};margin-left:3px;vertical-align:middle;flex-shrink:0" '
        f'title="{title}"></span>'
    )


# ──────────────────────────────────────────────────────────
#  Compactage des labels fins (strip hint_, abréviations)
# ──────────────────────────────────────────────────────────

_COMPACT_LABEL: dict[str, str] = {
    "hint_person_name":   "pers.name",
    "hint_person_role":   "pers.role",
    "hint_norp":          "norp",
    "hint_group_role":    "group.role",
    "hint_org_name":      "org",
    "hint_gpe":           "gpe",
    "hint_fac_name":      "facility",
    "hint_loc_generic":   "loc",
    "hint_infra":         "infra",
    "hint_weapon":        "weapon",
    "hint_vehicle":       "vehicle",
    "hint_substance":     "substance",
    "hint_food":          "food",
    "hint_tool":          "tool",
    "hint_object_generic":"object",
    "hint_object_name":   "product",
    "hint_event_nominal": "evt.nominal",
    "hint_event_named":   "evt.named",
    "hint_time_date":     "date",
    "hint_time_clock":    "clock",
    "hint_time_duration": "duration",
    "hint_quantity":      "qty",
    "hint_measure":       "measure",
    "hint_percentage":    "pct",
    "hint_count":         "count",
    "hint_money":         "money",
    "hint_rate":          "rate",
    "hint_law":           "law",
    "hint_work_of_art":   "art",
    "hint_concept":       "concept",
    "hint_disease":       "disease",
    "hint_language":      "lang",
}

def _compact(label: str) -> str:
    """Retourne le label compacté (sans hint_, abrégé si possible)."""
    return _COMPACT_LABEL.get(label, label.replace("hint_", ""))


def _span_html(text: str, bg: str, fg: str, label: str, data: dict, span_id: str) -> str:
    """Mark style Label Studio : texte surligné + label superscript compact."""
    data_json = _escape(json.dumps(data, ensure_ascii=False))
    display_label = _compact(label)

    # Badge override NER
    ner_badge = ""
    if data.get("ner_override"):
        _nr = _compact(data["ner_override"])
        ner_badge = f'<span class="pill-override" title="NER override: {data["ner_override"]}">🔗{_escape(_nr)}</span>'

    # Dot confiance (discret, inline après le label)
    score = data.get("score") or data.get("svo_prob") or data.get("svo_boundary_prob")
    if score is None or score >= 0.85:
        dot = ""
    elif score >= 0.60:
        dot = '<span class="pill-dot pill-dot-mid" title="conf {:.2f}"></span>'.format(score)
    else:
        dot = '<span class="pill-dot pill-dot-low" title="conf {:.2f}"></span>'.format(score)

    return (
        f'<mark id="{span_id}" class="ner-pill" '
        f'style="--pill-bg:{bg};--pill-fg:{fg};" '
        f'data-info="{data_json}" onclick="selectSpan(this)" title="{label}">'
        f'{_escape(text)}'
        f'<span class="pill-label">{_escape(display_label)}</span>'
        f'{ner_badge}{dot}'
        f'</mark>'
    )


_ALL_COARSE = [c for c in COARSE_LABELS if c != "NONE"]

# Priorité par défaut : named/spécifique avant nominal/générique, coarse en fallback.
# Construite à partir de COARSE_TO_FINE réel → toujours cohérente avec labels.py.
_FINE_PRIORITY_ORDER = [
    # EVENT : named avant nominal
    "hint_event_named", "hint_event_nominal",
    # PER : nom propre > norp/groupe > rôle générique
    "hint_person_name", "hint_norp", "hint_group_role", "hint_person_role",
    # LOC : toponymie précise > infrastructure > générique
    "hint_gpe", "hint_fac_name", "hint_infra", "hint_loc_generic",
    # ORG (un seul fine)
    "hint_org_name",
    # TIME : date/heure précise > durée
    "hint_time_date", "hint_time_clock", "hint_time_duration",
    # OBJECT : nommé > spécialisé > générique
    "hint_object_name", "hint_weapon", "hint_vehicle",
    "hint_food", "hint_substance", "hint_tool", "hint_object_generic",
    # VALUE : montant/mesure explicite > quantités vagues
    "hint_money", "hint_measure", "hint_percentage", "hint_rate",
    "hint_quantity", "hint_count",
    # ABSTRACT : légal/œuvre > maladie/langue > concept vague
    "hint_law", "hint_work_of_art", "hint_disease", "hint_language", "hint_concept",
]
# Vérification de cohérence au démarrage (évite de sourcer un label fantôme)
_FINE_PRIORITY_ORDER = [l for l in _FINE_PRIORITY_ORDER if l in FINE2ID]
DEFAULT_SPAN_PRIORITY = ", ".join(_FINE_PRIORITY_ORDER + _ALL_COARSE)


def _parse_priority(priority_str: str) -> dict[str, int]:
    """Convertit une chaîne 'A, B, C' en {A:0, B:1, C:2}."""
    labels = [l.strip() for l in priority_str.split(",") if l.strip()]
    return {lbl: rank for rank, lbl in enumerate(labels)}


def resolve_overlaps_by_priority(spans: list, priority_ranks: dict) -> list:
    """
    NMS avec priorité de label configurable.
    - priority_ranks: {label: rank} — rank bas = haute priorité
    - À priorité égale → span le plus long gagne
    - Les spans perdants strictement contenus dans le gagnant sont conservés
      dans winner["components"] plutôt qu'écrasés.
    """
    max_rank = len(priority_ranks) if priority_ranks else 0

    def rank_of(sp):
        return min(
            priority_ranks.get(sp.get("label", ""),  max_rank),
            priority_ranks.get(sp.get("coarse", ""), max_rank),
        )

    def _as_component(sp):
        d = sp.get("data", {})
        return {
            "text":       sp.get("text", ""),
            "char_start": sp["start"],
            "char_end":   sp["end"],
            "coarse":     sp.get("coarse", ""),
            "fine":       sp.get("label", sp.get("coarse", "")),
            "score":      d.get("score"),
        }

    sorted_spans = sorted(spans, key=lambda sp: (rank_of(sp), -(sp["end"] - sp["start"])))

    selected: list[dict] = []
    for sp in sorted_spans:
        overlapping = [sel for sel in selected
                       if sp["start"] < sel["end"] and sp["end"] > sel["start"]]
        if not overlapping:
            sp = dict(sp)
            sp.setdefault("components", [])
            selected.append(sp)
        else:
            # Strictement contenu dans un gagnant → enregistré comme composant
            for winner in overlapping:
                if sp["start"] >= winner["start"] and sp["end"] <= winner["end"]:
                    winner.setdefault("components", [])
                    winner["components"].append(_as_component(sp))

    selected.sort(key=lambda x: x["start"])
    return selected


def build_annotated_html(text: str, ner_spans: list, svo_spans: list,
                          show_svo: bool, show_arcs: bool = False,
                          sent_id: str = "s0",
                          fine_for_coarse: list | None = None,
                          priority_ranks: dict | None = None) -> str:
    """
    Reconstruit le texte avec les spans NER surlignés.
    fine_for_coarse : catégories coarse pour lesquelles on affiche le label FIN.
    priority_ranks  : {label: rank} issu de _parse_priority() — gère l'ordre de résolution des overlaps.
    Les spans SVO ne sont plus rendus inline — uniquement via arcs SVG (si show_arcs=True).
    """
    fine_set = set(fine_for_coarse) if fine_for_coarse is not None else set(_ALL_COARSE)
    p_ranks  = priority_ranks if priority_ranks is not None else {}

    # Construire les candidats NER
    all_spans = []
    for i, e in enumerate(ner_spans):
        coarse = e.get("coarse", "NONE")
        label = e.get("fine", coarse) if coarse in fine_set else coarse
        all_spans.append({
            "start":  e["char_start"], "end": e["char_end"],
            "text":   e["text"],
            "kind":   "ner",
            "label":  label,
            "coarse": coarse,
            "bg":     COARSE_COLORS.get(coarse, COARSE_COLORS["NONE"])[0],
            "fg":     COARSE_COLORS.get(coarse, COARSE_COLORS["NONE"])[1],
            "data":   {k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()},
            "id":     f"ner_{i}",
        })

    # Résolution des overlaps avec priorité configurable
    all_spans = resolve_overlaps_by_priority(all_spans, p_ranks)

    # Reconstruction du HTML
    html_parts = []
    cursor = 0
    for sp in all_spans:
        s, e = sp["start"], sp["end"]
        if s > cursor:
            html_parts.append(_escape(text[cursor:s]))
        data_with_comp = dict(sp["data"])
        if sp.get("components"):
            data_with_comp["_components"] = sp["components"]
        html_parts.append(_span_html(sp["text"], sp["bg"], sp["fg"],
                                      sp["label"], data_with_comp, sp["id"]))
        cursor = e

    if cursor < len(text):
        html_parts.append(_escape(text[cursor:]))

    annotated = "".join(html_parts)
    if show_arcs and show_svo and svo_spans:
        arc_svg = build_triplet_arc_svg(text, svo_spans, sent_id=sent_id)
        return annotated + arc_svg
    return annotated


# ──────────────────────────────────────────────────────────
#  Réconciliation NER ↔ SVO
# ──────────────────────────────────────────────────────────

# Coarse NER compatibles avec chaque rôle SVO
_SUBJ_COARSE = {"PER", "ORG", "EVENT", "ABSTRACT"}
_OBJ_COARSE  = {"PER", "ORG", "LOC", "EVENT", "OBJECT", "ABSTRACT", "VALUE", "TIME"}

def reconcile_svo_with_ner(
    svo_spans: list,
    ner_spans: list,
    min_ner_score: float = 0.50,
) -> list:
    """
    Pour chaque span SVO sujet/objet :
      • Si un span NER l'enveloppe (NER plus large) → étend le texte/positions du span SVO
      • Marque le span avec 'ner_override' pour l'affichage
    Renvoie la liste SVO enrichie (inchangée pour verbes/iobj).
    """
    subject_roles = {"svo_subject", "pron_subj"}
    object_roles  = {"svo_object", "pron_obj", "svo_iobj"}
    enriched = []

    for s in svo_spans:
        role = s.get("svo_role", "")
        if role not in subject_roles | object_roles:
            enriched.append(s)
            continue

        s_start, s_end = s["char_start"], s["char_end"]
        allowed_coarse = _SUBJ_COARSE if role in subject_roles else _OBJ_COARSE

        best_ner = None
        best_score = 0.0
        for e in ner_spans:
            if e.get("score", 0) < min_ner_score:
                continue
            if e.get("coarse") not in allowed_coarse:
                continue
            n_start, n_end = e["char_start"], e["char_end"]
            # overlap > 0
            if min(s_end, n_end) - max(s_start, n_start) <= 0:
                continue
            # NER doit envelopper ou être plus grand
            if n_end - n_start < s_end - s_start:
                continue
            if e["score"] > best_score:
                best_ner = e
                best_score = e["score"]

        if best_ner and (
            best_ner["char_start"] <= s_start
            and best_ner["char_end"] >= s_end
        ):
            s = {
                **s,
                "text":       best_ner["text"],
                "char_start": best_ner["char_start"],
                "char_end":   best_ner["char_end"],
                "ner_override":      best_ner.get("fine", best_ner.get("coarse")),
                "ner_override_score": round(best_ner["score"], 4),
            }

        enriched.append(s)

    return enriched


def fill_null_subjects(
    svo_spans: list,
    ner_spans: list,
    max_gap_chars: int = 120,
    min_ner_score:  float = 0.60,
) -> list:
    """
    Pour chaque verbe SVO sans aucun sujet à gauche, cherche l'entité NER
    PER/ORG la plus proche à gauche et crée un span svo_subject synthétique.
    Renvoie la liste SVO augmentée.
    """
    verbs    = [s for s in svo_spans if s.get("svo_role") == "svo_verb"]
    subjects = [s for s in svo_spans if s.get("svo_role") in ("svo_subject", "pron_subj")]
    extra    = []

    for v in verbs:
        v_start = v["char_start"]
        # Déjà un sujet à gauche ?
        if any(s["char_end"] <= v_start for s in subjects):
            continue

        candidates = [
            e for e in ner_spans
            if e.get("coarse") in _SUBJ_COARSE
            and e["char_end"] <= v_start
            and v_start - e["char_end"] <= max_gap_chars
            and e.get("score", 0) >= min_ner_score
        ]
        if not candidates:
            continue

        best = max(candidates, key=lambda x: x["char_end"])
        extra.append({
            "text":       best["text"],
            "svo_role":   "svo_subject",
            "char_start": best["char_start"],
            "char_end":   best["char_end"],
            "voice":      v.get("voice", "ACTIVE"),
            "gender":     best.get("gender"),
            "number":     best.get("number"),
            "svo_boundary_prob": 0.0,
            "svo_prob":          0.0,
            "voice_prob":        0.0,
            "from_ner":          True,
            "ner_override":      best.get("fine", best.get("coarse")),
            "ner_override_score": round(best["score"], 4),
        })

    return svo_spans + extra


def build_triplet_arc_svg(text: str, svo_spans: list, sent_id: str = "s0") -> str:
    """
    SVG displaCy-style.
    • Tokens espacés uniformément (pas proportionnel aux char offsets)
      → labels jamais écrasés horizontalement.
    • Level-assignment par interval scheduling
      → arcs jamais écrasés verticalement.
    """
    relevant = sorted(
        [s for s in svo_spans if s.get("svo_role") in
         ("svo_verb", "svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj")],
        key=lambda s: s["char_start"],
    )
    if not relevant:
        return ""

    ROLE_COLOR = {
        "svo_verb":    "#0369a1",
        "svo_subject": "#15803d",
        "svo_object":  "#9d174d",
        "svo_iobj":    "#c2410c",
        "pron_subj":   "#166534",
        "pron_obj":    "#7e22ce",
    }
    ARC_LABEL = {
        "svo_subject": "nsubj", "pron_subj": "nsubj",
        "svo_object":  "dobj",  "pron_obj":  "dobj",
        "svo_iobj":    "iobj",
    }

    def mid_id(role):
        return f"arh-{sent_id}-{role.replace('_','')}"

    # ── 1. Espacement uniforme ─────────────────────────────────────────────
    PAD_X      = 40
    TOK_SPACE  = 130          # px entre centres de tokens
    TOKEN_Y    = 26           # ligne des labels texte
    BASE_Y     = TOKEN_Y + 22 # ligne d'attache des arcs
    LEVEL_H    = 26           # hauteur par niveau d'arc
    ARC_BASE_H = 16           # hauteur minimale (niveau 0)
    FONT_SIZE  = 13

    n = len(relevant)
    W = PAD_X * 2 + (n - 1) * TOK_SPACE
    W = max(W, 400)

    enriched = []
    for i, s in enumerate(relevant):
        enriched.append({
            **s,
            "_x": PAD_X + i * TOK_SPACE,
            "_c": ROLE_COLOR.get(s["svo_role"], "#64748b"),
            "_i": i,
        })

    # ── 2. Collecte des arcs ───────────────────────────────────────────────
    subject_roles = {"svo_subject", "pron_subj"}
    object_roles  = {"svo_object", "svo_iobj", "pron_obj"}
    verbs = [s for s in enriched if s["svo_role"] == "svo_verb"]

    arc_list = []
    for v in verbs:
        subjs = sorted(
            [s for s in enriched if s["svo_role"] in subject_roles
             and s["char_end"] <= v["char_start"]],
            key=lambda s: -s["char_end"],
        )[:1]
        objs = sorted(
            [s for s in enriched if s["svo_role"] in object_roles
             and s["char_start"] >= v["char_end"]],
            key=lambda s: s["char_start"],
        )[:2]
        for dep in subjs + objs:
            is_subj = dep in subjs
            x1 = dep["_x"] if is_subj else v["_x"]
            x2 = v["_x"]   if is_subj else dep["_x"]
            arc_list.append({
                "x1":    x1,    "x2":  x2,
                "xi1":   min(dep["_i"], v["_i"]),
                "xi2":   max(dep["_i"], v["_i"]),
                "color": dep["_c"],
                "lbl":   ARC_LABEL.get(dep["svo_role"], "dep"),
                "marker": mid_id(dep["svo_role"]),
            })

    # ── 3. Level-assignment sur indices de tokens (plus stable que px) ────
    arc_list.sort(key=lambda a: a["xi2"] - a["xi1"])  # courts d'abord
    occupied: list[tuple[int, int, int]] = []
    for arc in arc_list:
        level = 0
        while any(
            arc["xi1"] < ox2 and arc["xi2"] > ox1 and ol == level
            for ox1, ox2, ol in occupied
        ):
            level += 1
        arc["_level"] = level
        occupied.append((arc["xi1"], arc["xi2"], level))

    max_level = max((a["_level"] for a in arc_list), default=0)
    SVG_H = BASE_Y + ARC_BASE_H + (max_level + 1) * LEVEL_H + 20

    # ── 4. Génération SVG ─────────────────────────────────────────────────
    defs = ["<defs>"]
    # Drop-shadow filter
    defs.append(
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#00000014"/>'
        '</filter>'
    )
    for role, color in ROLE_COLOR.items():
        if role == "svo_verb":
            continue
        defs.append(
            f'<marker id="{mid_id(role)}" markerWidth="7" markerHeight="7" '
            f'refX="5" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L7,3.5 L0,7 L2,3.5 Z" fill="{color}"/>'
            f'</marker>'
        )
    defs.append("</defs>")

    parts = ["".join(defs)]
    # Fond blanc avec ombre légère
    parts.append(
        f'<rect width="{W}" height="{SVG_H}" rx="10" '
        f'fill="white" stroke="#e2e8f0" stroke-width="1" filter="url(#shadow)"/>'
    )
    # Header label "Dépendances SVO"
    parts.append(
        f'<text x="10" y="13" font-family="Inter,system-ui" font-size="8.5" '
        f'fill="#94a3b8" font-weight="600" letter-spacing="0.08em" text-anchor="start">'
        f'DÉPENDANCES SVO</text>'
    )

    # Tokens : pill (rect arrondi) + texte
    ROK_BG = {  # fond pastille par rôle
        "svo_verb":    "#e0f2fe", "svo_subject": "#dcfce7",
        "svo_object":  "#fce7f3", "svo_iobj":    "#fff7ed",
        "pron_subj":   "#f0fdf4", "pron_obj":    "#fdf4ff",
    }
    for sp in enriched:
        x, color = sp["_x"], sp["_c"]
        pill_bg = ROK_BG.get(sp["svo_role"], "#f1f5f9")
        emoji = SVO_EMOJI.get(sp["svo_role"], "")
        raw = sp["text"]
        lbl = raw[:13] + "…" if len(raw) > 13 else raw
        full = emoji + " " + lbl if emoji else lbl
        pill_w = max(len(full) * 7.8 + 20, 44)
        parts.append(
            f'<rect x="{x - pill_w/2:.1f}" y="{TOKEN_Y - 17}" '
            f'width="{pill_w:.1f}" height="21" rx="10" '
            f'fill="{pill_bg}" stroke="{color}38" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x}" y="{TOKEN_Y - 3}" text-anchor="middle" '
            f'font-family="Inter,system-ui" font-size="12" '
            f'fill="{color}" font-weight="600">{_escape(full)}</text>'
        )
        parts.append(
            f'<line x1="{x}" y1="{TOKEN_Y + 4}" x2="{x}" y2="{BASE_Y}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="2,2" opacity="0.4"/>'
        )

    # Arcs
    for arc in arc_list:
        x1, x2 = arc["x1"], arc["x2"]
        mid_x  = (x1 + x2) / 2
        h      = ARC_BASE_H + arc["_level"] * LEVEL_H
        ctrl_y = BASE_Y + h
        color  = arc["color"]
        # Tracé de l'arc
        parts.append(
            f'<path d="M{x1},{BASE_Y} Q{mid_x:.1f},{ctrl_y:.1f} {x2},{BASE_Y}" '
            f'fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="5,3" '
            f'stroke-linecap="round" marker-end="url(#{arc["marker"]})"/>'
        )
        # Bubble label (petit rect + texte)
        lbl_text = arc["lbl"]
        lbl_w = len(lbl_text) * 6.2 + 10
        lbl_x = mid_x - lbl_w / 2
        lbl_y = ctrl_y + 6
        parts.append(
            f'<rect x="{lbl_x:.1f}" y="{lbl_y:.1f}" '
            f'width="{lbl_w:.1f}" height="13" rx="4" '
            f'fill="white" stroke="{color}50" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{mid_x:.1f}" y="{lbl_y + 9.5:.1f}" text-anchor="middle" '
            f'font-family="Inter,system-ui" font-size="8.5" fill="{color}" '
            f'font-weight="600" letter-spacing="0.03em">{lbl_text}</text>'
        )

    return (
        f'<div class="svo-arc-wrap" style="margin-top:4px;overflow-x:auto;border-radius:5px">'
        f'<svg viewBox="0 0 {W} {SVG_H}" '
        f'style="width:100%;max-width:{W}px;display:block;height:{SVG_H}px">'
        + "\n".join(parts)
        + "</svg></div>"
    )



def build_legend_html() -> str:
    items = ['<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0 6px;font-family:Inter,system-ui,sans-serif">']
    items.append('<span style="font-size:.72em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-right:4px">NER</span>')
    for coarse, (bg, fg) in COARSE_COLORS.items():
        if coarse == "NONE":
            continue
        items.append(
            f'<span style="background:{bg};color:{fg};border-radius:6px;'
            f'padding:2px 9px;font-size:.78em;font-weight:600;'
            f'border:1.5px solid {fg}28;box-shadow:0 1px 3px {fg}18">'
            f'{coarse}</span>'
        )
    items.append('<span style="font-size:.72em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-left:8px;margin-right:4px">SVO arcs</span>')
    arc_legend = [
        ("nsubj", "#15803d", "#dcfce7"),
        ("verb",  "#0369a1", "#e0f2fe"),
        ("dobj",  "#9d174d", "#fce7f3"),
        ("iobj",  "#c2410c", "#fff7ed"),
    ]
    for lbl, fg, bg in arc_legend:
        items.append(
            f'<span style="background:{bg};color:{fg};border-radius:6px;'
            f'padding:2px 9px;font-size:.78em;font-weight:600;'
            f'border:1.5px solid {fg}28;box-shadow:0 1px 3px {fg}18">'
            f'{lbl}</span>'
        )
    items.append('</div>')
    return "".join(items)

# ──────────────────────────────────────────────────────────
#  JS intégré — handleClick → met à jour le champ hidden
# ──────────────────────────────────────────────────────────

JS_CLICK = """
<script>
/* Ré-injecté à chaque mise à jour du résultat HTML via createElement — exécution garantie */

function _dfEsc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function _dfRow(k,v){ return '<tr><td>'+k+'</td><td>'+v+'</td></tr>'; }
function _dfCode(v){ return v!=null?'<code>'+_dfEsc(v)+'</code>':'<span style="opacity:.4">—</span>'; }

function _dfRender(d){
    if(!d) return '<p style="color:#94a3b8;margin:0">Sélectionnez un span.</p>';
    var rows='', title='';
    if(d.coarse!==undefined && d.fine!==undefined){
        var lbl=(d.fine&&d.fine!=='NONE')?d.fine:d.coarse;
        title='🏷 <code>'+_dfEsc(lbl)+'</code>';
        rows+=_dfRow('Texte',   _dfCode(d.text));
        rows+=_dfRow('Coarse',  _dfCode(d.coarse));
        rows+=_dfRow('Fine',    _dfCode(d.fine));
        rows+=_dfRow('Chars',   '['+d.char_start+':'+d.char_end+']');
        rows+=_dfRow('Tokens',  '['+d.tok_start+':'+d.tok_end+']');
        rows+='<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.82em;color:#64748b;letter-spacing:.05em">SCORES</td></tr>';
        rows+=_dfRow('p_entity',  d.boundary_prob!=null?d.boundary_prob:'—');
        rows+=_dfRow('p_coarse',  d.coarse_prob!=null?d.coarse_prob:'—');
        rows+=_dfRow('p_fine',    d.fine_prob!=null?d.fine_prob:'—');
        rows+=_dfRow('score', '<strong>'+(d.score!=null?d.score:'—')+'</strong>');
        if(d.ner_override) rows+=_dfRow('🔗 override',_dfCode(d.ner_override)+' ('+d.ner_override_score+')');
        if(d._components && d._components.length){
            rows+='<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.82em;color:#64748b;letter-spacing:.05em">COMPOSANTS ('+d._components.length+')</td></tr>';
            d._components.forEach(function(c,ci){
                rows+='<tr><td style="padding-left:10px;color:#94a3b8;font-size:.88em">#'+(ci+1)+'</td>'
                     +'<td style="font-size:.88em"><code>'+_dfEsc(c.fine||c.coarse)+'</code> '
                     +_dfCode(c.text)
                     +(c.score!=null?' <span style="opacity:.55;font-size:.9em">('+c.score+')</span>':'')
                     +'</td></tr>';
            });
        }
    } else if(d.svo_role){
        var em={svo_verb:'🔵',svo_subject:'🟢',svo_object:'🔴',svo_iobj:'🟠',pron_subj:'🟢',pron_obj:'🔴'};
        title=(em[d.svo_role]||'⚪')+' <code>'+_dfEsc(d.svo_role)+'</code>';
        rows+=_dfRow('Texte',   _dfCode(d.text));
        rows+=_dfRow('Rôle',    _dfCode(d.svo_role));
        rows+=_dfRow('Voice',   _dfCode(d.voice));
        rows+=_dfRow('Chars',   '['+d.char_start+':'+d.char_end+']');
        if(d.from_ner) rows+=_dfRow('Source','<em>synthétique NER</em>');
        if(d.ner_override) rows+=_dfRow('🔗 override',_dfCode(d.ner_override)+' ('+d.ner_override_score+')');
        rows+='<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.82em;color:#64748b;letter-spacing:.05em">SCORES</td></tr>';
        rows+=_dfRow('p_boundary', d.svo_boundary_prob!=null?d.svo_boundary_prob:'—');
        rows+=_dfRow('p_role',     d.svo_prob!=null?d.svo_prob:'—');
        rows+=_dfRow('voice conf', d.voice_prob!=null?d.voice_prob:'—');
        if(d.gender||d.number) rows+=_dfRow('genre/nb',_dfEsc((d.gender||'—')+' / '+(d.number||'—')));
    } else {
        title='Données';
        Object.keys(d).forEach(function(k){ rows+=_dfRow(_dfEsc(k),_dfCode(d[k])); });
    }
    return '<h4 style="margin:0 0 10px;padding-right:22px">'+title+'</h4><table>'+rows+'</table>';
}

function _dfGetPanel(){
    var p=document.getElementById('ner-detail-float');
    if(p) return p;
    p=document.createElement('div');
    p.id='ner-detail-float';
    p.innerHTML='<button class="df-close" onclick="document.getElementById(\'ner-detail-float\').style.display=\'none\'">✕</button>'
               +'<div id="ner-detail-content"></div>';
    document.body.appendChild(p);
    document.addEventListener('click',function(e){
        if(!p.contains(e.target)&&!e.target.closest('mark.ner-pill')){
            p.style.display='none';
            document.querySelectorAll('mark.ner-pill.selected').forEach(function(m){m.classList.remove('selected');});
        }
    });
    return p;
}

window.selectSpan=function(el){
    document.querySelectorAll('mark.ner-pill.selected').forEach(function(m){m.classList.remove('selected');});
    el.classList.add('selected');
    var d=null;
    try{d=JSON.parse(el.getAttribute('data-info'));}catch(e){}
    var panel=_dfGetPanel();
    var c=document.getElementById('ner-detail-content');
    if(c) c.innerHTML=_dfRender(d);
    var rect=el.getBoundingClientRect();
    panel.style.top=Math.min(rect.bottom+window.scrollY+6, window.innerHeight+window.scrollY-340)+'px';
    panel.style.display='block';
};

window.toggleDark=function(){
    var html=document.documentElement;
    var isDark=html.getAttribute('data-theme')!=='dark';
    html.setAttribute('data-theme',isDark?'dark':'light');
    localStorage.setItem('ner-dark',isDark?'1':'0');
    var btn=document.getElementById('dark-toggle');
    if(btn) btn.textContent=isDark?'☀️':'🌙';
};

(function applyDark(){
    if(localStorage.getItem('ner-dark')==='1'){
        document.documentElement.setAttribute('data-theme','dark');
        var btn=document.getElementById('dark-toggle');
        if(btn){btn.textContent='☀️';}
        else{setTimeout(applyDark,200);}
    }
})();
</script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.ner-result-wrap {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.02em;
    line-height: 2.2;
    color: #1e293b;
}

/* ── Label Studio style : texte + label inline dans le même fond ── */
mark.ner-pill {
    background: var(--pill-bg);
    color: var(--pill-fg);
    border-radius: 4px;
    padding: 2px 6px 2px 5px;
    margin: 0 2px;
    cursor: pointer;
    display: inline;
    text-decoration: none;
    font-weight: 500;
    transition: filter 0.12s, box-shadow 0.12s;
    outline: 2px solid transparent;
    outline-offset: 1px;
}
mark.ner-pill:hover   { filter: brightness(0.91); }
mark.ner-pill.selected {
    outline: 2px solid #f59e0b;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.20);
}

/* ── Label inline séparé par un micro-trait vertical ── */
span.pill-label {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-left: 5px;
    padding-left: 5px;
    border-left: 1.5px solid var(--pill-fg);
    opacity: 0.75;
    vertical-align: baseline;
}

.pill-override {
    font-size: 0.65em;
    margin-left: 3px;
    opacity: 0.65;
}

/* ── Cartes de phrases ── */
.sent-card {
    background: var(--card, white);
    border-radius: 10px;
    padding: 14px 18px 12px;
    margin-bottom: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid var(--card-border, #f1f5f9);
}
.sent-num {
    display: inline-block;
    font-size: 0.66em; font-weight: 700;
    color: var(--text-muted, #94a3b8);
    background: var(--border, #f1f5f9);
    border-radius: 4px;
    padding: 1px 5px; margin-right: 8px;
    vertical-align: middle;
}

/* ── Panneau flottant détail ── */
#ner-detail-float {
    position: fixed;
    right: 18px; top: 76px;
    width: 310px;
    max-height: 72vh;
    overflow-y: auto;
    background: #fff;
    border-radius: 12px;
    padding: 14px 16px 16px;
    box-shadow: 0 8px 36px rgba(0,0,0,0.18);
    z-index: 9999;
    display: none;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.84em;
    border: 1px solid #e2e8f0;
}
html[data-theme="dark"] #ner-detail-float {
    background: #1e293b;
    border-color: #334155;
    color: #e2e8f0;
}
#ner-detail-float .df-close {
    position: absolute; top: 8px; right: 10px;
    background: none; border: none;
    font-size: 1.1em; cursor: pointer;
    color: #94a3b8; line-height: 1;
    padding: 2px 5px; border-radius: 4px;
}
#ner-detail-float .df-close:hover { background: #f1f5f9; color: #1e293b; }
#ner-detail-float h4 { margin: 0 0 10px; font-size: 1em; }
#ner-detail-float table { width: 100%; border-collapse: collapse; }
#ner-detail-float td {
    padding: 3px 6px; vertical-align: top;
    border-bottom: 1px solid #f1f5f9; font-size: 0.93em;
}
#ner-detail-float td:first-child {
    color: #64748b; white-space: nowrap;
    font-weight: 600; width: 42%;
}
html[data-theme="dark"] #ner-detail-float td { border-color: #334155; }
html[data-theme="dark"] #ner-detail-float td:first-child { color: #94a3b8; }
#ner-detail-float code {
    background: #f1f5f9; border-radius: 3px;
    padding: 1px 5px; font-size: 0.92em;
    word-break: break-all;
}
html[data-theme="dark"] #ner-detail-float code { background: #0f172a; color: #93c5fd; }
</style>
"""

# ──────────────────────────────────────────────────────────
#  Formatage du panneau détails
# ──────────────────────────────────────────────────────────

def format_details(info_json: str) -> str:
    if not info_json or info_json.strip() in ("", "null"):
        return "*Cliquez sur un span surligné pour voir ses détails.*"

    try:
        d = json.loads(info_json)
    except Exception:
        return f"(JSON invalide : {info_json})"

    lines = []

    # Distinguer NER vs SVO
    if "coarse" in d and "fine" in d:
        # NER
        lines.append(f"## 🏷 NER — `{d.get('fine', '?')}`")
        lines.append(f"**Texte** : `{d.get('text', '?')}`")
        lines.append(f"**Coarse** : `{d.get('coarse', '?')}`  |  **Fine** : `{d.get('fine', '?')}`")
        lines.append(f"**Positions** : char [{d.get('char_start')} : {d.get('char_end')}]  |  tok [{d.get('tok_start')} : {d.get('tok_end')}]")
        lines.append("")
        lines.append("### Scores")
        lines.append(f"| Métrique | Valeur |")
        lines.append(f"|---|---|")
        lines.append(f"| p_entity (boundary) | `{d.get('boundary_prob', '?')}` |")
        lines.append(f"| p_coarse            | `{d.get('coarse_prob', '?')}` |")
        lines.append(f"| p_fine              | `{d.get('fine_prob', '?')}` |")
        lines.append(f"| **score global**    | **`{d.get('score', '?')}`** |")

    elif "svo_role" in d:
        # SVO
        role = d.get("svo_role", "?")
        emoji = SVO_EMOJI.get(role, "⚪")
        lines.append(f"## {emoji} SVO — `{role}`")
        lines.append(f"**Texte** : `{d.get('text', '?')}`")
        lines.append(f"**Rôle** : `{role}`  |  **Voice** : `{d.get('voice', '?')}`")
        lines.append(f"**Positions** : char [{d.get('char_start')} : {d.get('char_end')}]  |  tok [{d.get('tok_start', '?')} : {d.get('tok_end', '?')}]")
        if d.get("ner_override"):
            lines.append(f"\n🔗 **NER override** : `{d['ner_override']}` (score `{d.get('ner_override_score', '?')}`)")
        if d.get("from_ner"):
            lines.append("⚠️ *Sujet synthétique issu de NER (aucun sujet SVO détecté)*")
        lines.append("")
        lines.append("### Scores & morphologie")
        lines.append("| Métrique | Valeur |")
        lines.append("|---|---|")
        lines.append(f"| p_svo_boundary | `{d.get('svo_boundary_prob', '?')}` |")
        lines.append(f"| p_role         | `{d.get('svo_prob', '?')}` |")
        lines.append(f"| voice          | `{d.get('voice', '?')}` (conf `{d.get('voice_prob', '?')}`) |")
        g = d.get("gender")
        n = d.get("number")
        if g or n:
            lines.append(f"| genre / nombre | `{g or '—'}` / `{n or '—'}` |")
    else:
        lines.append("### Données brutes")
        for k, v in d.items():
            lines.append(f"- **{k}** : `{v}`")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
#  Fonction analyse principale (streaming)
# ──────────────────────────────────────────────────────────

def analyse(text_input: str, show_svo: bool, post_process: bool,
            tau_boundary: float, tau_svo: float, auto_split: bool,
            do_reconcile: bool = True,
            tau_none: float = 0.99, tau_coarse: float = 0.00, tau_fine: float = 0.00,
            min_ner_score_reconcile: float = 0.50, min_ner_score_fill: float = 0.60,
            max_gap_chars: int = 120,
            show_arcs: bool = True,
            fine_for_coarse: list | None = None,
            span_priority: str = DEFAULT_SPAN_PRIORITY):
    if not text_input.strip():
        yield "<i>Entrez du texte ci-dessus.</i>", "", "*Aucun résultat.*", gr.DownloadButton(visible=False)
        return

    priority_ranks = _parse_priority(span_priority)

    model, tokenizer, device = get_model()

    # ── Découpage en phrases ───────────────────────────────────────────────
    if auto_split:
        # Double saut de ligne = limite de paragraphe (frontière dure)
        # Saut de ligne simple = espace (ignoré)
        phrases = []
        paragraphs = re.split(r'\n{2,}', text_input.strip())
        for para in paragraphs:
            normalized = ' '.join(l.strip() for l in para.splitlines() if l.strip())
            if normalized:
                phrases.extend(split_sentences(normalized))
    else:
        phrases = [l.strip() for l in text_input.strip().splitlines() if l.strip()]

    if not phrases:
        yield "<i>Aucune phrase exploitable.</i>", "", "*Aucun résultat.*", gr.DownloadButton(visible=False)
        return

    html_blocks = []
    stats_lines = []
    total_ner = 0
    total_svo = 0
    json_records = []
    t0_infer = time.perf_counter()

    # Affiche immédiatement le texte brut (skeleton)
    pending_blocks = [
        f'<div class="sent-card" style="opacity:.5">'
        f'<span class="sent-num">#{i+1}</span>'
        f'<span class="ner-result-wrap" style="color:#94a3b8">{_escape(p)}</span>'
        f'</div>'
        for i, p in enumerate(phrases)
    ]
    initial_html = (
        JS_CLICK + build_legend_html()
        + '<div class="ner-result-wrap" style="padding:4px 0">'
        + "\n".join(pending_blocks) + "</div>"
    )
    yield initial_html, "*Inférence en cours…*", "*Cliquez sur un span une fois les labels affichés.*", gr.DownloadButton(visible=False)

    # Streaming par mini-batch pour compromis fluidité/perf
    chunk_size = 4
    processed = 0

    for chunk_start in range(0, len(phrases), chunk_size):
        chunk_texts = phrases[chunk_start:chunk_start + chunk_size]
        chunk_results = predict_texts_batch(
            model=model, tokenizer=tokenizer, texts=chunk_texts, device=device,
            max_length=128, max_span_len=12,
            tau_boundary=tau_boundary, tau_none=tau_none, tau_coarse=tau_coarse,
            tau_fine=tau_fine, topk_coarse=2, min_char_len=2,
            enforce_word_boundaries=True, tau_svo_boundary=tau_svo,
        )

        for local_i, (phrase, res) in enumerate(zip(chunk_texts, chunk_results)):
            i = chunk_start + local_i
            ner = res["ner"]
            svo = res["svo"]
            if post_process:
                ner = post_process_dynamic(ner)
            else:
                ner = dedupe_overlaps(ner)

            # ── Réconciliation NER ↔ SVO ─────────────────────────────
            if do_reconcile and show_svo:
                svo = reconcile_svo_with_ner(svo, ner, min_ner_score=min_ner_score_reconcile)
                svo = fill_null_subjects(svo, ner, max_gap_chars=max_gap_chars, min_ner_score=min_ner_score_fill)

            total_ner += len(ner)
            total_svo += len(svo)

            block = build_annotated_html(
                phrase, ner, svo if show_svo else [], show_svo,
                show_arcs=show_arcs, sent_id=f"s{i}",
                fine_for_coarse=fine_for_coarse,
                priority_ranks=priority_ranks,
            )
            if len(phrases) > 1:
                html_blocks.append(
                    f'<div class="sent-card">'
                    f'<span class="sent-num">#{i+1}</span>'
                    f'<span class="ner-result-wrap">{block}</span>'
                    f'</div>'
                )
            else:
                html_blocks.append(
                    f'<div class="sent-card"><span class="ner-result-wrap">{block}</span></div>'
                )

            coarse_counts = {}
            for e in ner:
                coarse = e.get("coarse", "?")
                coarse_counts[coarse] = coarse_counts.get(coarse, 0) + 1
            svo_counts = {}
            for s in svo:
                role = s.get("svo_role", "?")
                svo_counts[role] = svo_counts.get(role, 0) + 1

            stats_lines.append(f"**#{i+1}** — NER: {len(ner)} | SVO: {len(svo)}")
            for c, n in sorted(coarse_counts.items()):
                stats_lines.append(f"  - {c}: {n}")
            for r, n in sorted(svo_counts.items()):
                stats_lines.append(f"  - {SVO_EMOJI.get(r, '')} {r}: {n}")

            verbs = [s for s in svo if s.get("svo_role") == "svo_verb"]
            subjects = [s for s in svo if s.get("svo_role") in ("svo_subject", "pron_subj")]
            objects = [s for s in svo if s.get("svo_role") in ("svo_object", "svo_iobj", "pron_obj")]
            triplets = []
            for v in verbs:
                subj = max([s for s in subjects if s["char_end"] <= v["char_start"]],
                           key=lambda x: x["char_start"], default=None)
                obj = min([o for o in objects if o["char_start"] >= v["char_end"]],
                          key=lambda x: x["char_start"], default=None)
                triplets.append({
                    "subject": subj["text"] if subj else None,
                    "verb": v["text"],
                    "object": obj["text"] if obj else None,
                    "voice": v.get("voice"),
                })

            json_records.append({"idx": i + 1, "text": phrase, "ner": ner, "svo": svo, "triplets": triplets})
            processed += 1

        # Rendu progressif
        pending_html = [
            f'<div class="sent-card" style="opacity:.35">'
            f'<span class="sent-num">#{j+1}</span>'
            f'<span class="ner-result-wrap" style="color:#94a3b8">{_escape(phrases[j])}</span>'
            f'</div>'
            for j in range(processed, len(phrases))
        ]
        main_html = (
            JS_CLICK + build_legend_html()
            + '<div style="padding:4px 0">'
            + "\n".join(html_blocks + pending_html) + "</div>"
        )

        elapsed = time.perf_counter() - t0_infer
        stats_md = (
            f"**{len(phrases)} phrase(s)** — **NER : {total_ner}** | **SVO : {total_svo}**  "
            f"⏱ `{elapsed*1000:.0f}ms` total · `{elapsed/max(1, processed)*1000:.1f}ms`/phrase traitée  "
            f"(`{processed}/{len(phrases)}`)\n\n"
            + "\n".join(stats_lines)
        )

        yield main_html, stats_md, "", gr.DownloadButton(visible=False)

    # Export JSON final (fin de stream)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="ner_svo_", delete=False, encoding="utf-8"
    )
    json.dump(json_records, tmp, ensure_ascii=False, indent=2)
    tmp.close()

    final_elapsed = time.perf_counter() - t0_infer
    final_stats = (
        f"**{len(phrases)} phrase(s)** — **NER : {total_ner}** | **SVO : {total_svo}**  "
        f"⏱ `{final_elapsed*1000:.0f}ms` total · `{final_elapsed/len(phrases)*1000:.1f}ms`/phrase\n\n"
        + "\n".join(stats_lines)
    )

    final_html = (
        JS_CLICK + build_legend_html()
        + '<div style="padding:4px 0">'
        + "\n".join(html_blocks) + "</div>"
    )

    yield final_html, final_stats, "", gr.DownloadButton(label="⬇ Télécharger JSON", value=tmp.name, visible=True)

# ──────────────────────────────────────────────────────────
#  UI Gradio
# ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["Emmanuel Macron s'est rendu à Berlin pour rencontrer Olaf Scholz.", True, False, 0.40, 0.50, True],
    ["La Banque centrale européenne a relevé ses taux d'intérêt de 25 points de base mardi. Le président de la BCE, Mario Draghi, a justifié cette décision par la hausse de l'inflation.", True, False, 0.40, 0.50, True],
    ["Apple a annoncé le lancement de l'iPhone 17 le 15 septembre 2025 à Cupertino. Tesla a livré 500 000 véhicules au troisième trimestre. Google a racheté la start-up française Nabla pour 400 millions d'euros.", True, False, 0.40, 0.50, True],
    ["Le Conseil constitutionnel a censuré plusieurs articles de la loi immigration. La directive européenne sur l'intelligence artificielle entre en vigueur le 1er août. Le règlement impose des sanctions pouvant atteindre 4 % du chiffre d'affaires mondial.", True, True, 0.40, 0.50, True],
]

# ──────────────────────────────────────────────────────────
#  CSS moderne
# ──────────────────────────────────────────────────────────

CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Variables (light) ── */
:root {
    --bg:         #f0f2f5;
    --card:       #ffffff;
    --card-border:#f1f5f9;
    --text:       #1e293b;
    --text-muted: #94a3b8;
    --border:     #e2e8f0;
    --input-bg:   #ffffff;
    --acc-body:   #fafbff;
    --tab-sel-bg: #eff6ff;
    --tab-sel-fg: #1d4ed8;
    --chk-sel-bg: #eff6ff;
    --chk-sel-fg: #1d4ed8;
    --scrollbar:  #cbd5e1;
    --shadow-sm:  0 2px 12px rgba(0,0,0,0.07);
    --shadow-xs:  0 2px 8px rgba(0,0,0,0.06);
}

/* ── Variables (dark) ── */
:root[data-theme="dark"],
html[data-theme="dark"] {
    --bg:         #0f172a;
    --card:       #1e293b;
    --card-border:#334155;
    --text:       #e2e8f0;
    --text-muted: #64748b;
    --border:     #334155;
    --input-bg:   #0f172a;
    --acc-body:   #1a2640;
    --tab-sel-bg: #1e3a5f;
    --tab-sel-fg: #93c5fd;
    --chk-sel-bg: #1e3a5f;
    --chk-sel-fg: #93c5fd;
    --scrollbar:  #334155;
    --shadow-sm:  0 2px 12px rgba(0,0,0,0.35);
    --shadow-xs:  0 2px 8px rgba(0,0,0,0.30);
}

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    transition: background 0.25s, color 0.25s;
}

/* Masquer le champ bridge JS→Python tout en le gardant dans le DOM */
#span_click_data, .span-click-data {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    opacity: 0 !important; pointer-events: none !important;
    overflow: hidden !important; clip: rect(0,0,0,0) !important;
    white-space: nowrap !important;
}

.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Header hero ── */
#header-hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 60%, #6366f1 100%);
    border-radius: 16px;
    padding: 24px 36px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(29,78,216,0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
#header-hero-text h1 { color: white !important; font-size: 1.7em; margin: 0 0 4px; }
#header-hero-text p  { color: rgba(255,255,255,0.80) !important; margin: 0; font-size: 0.93em; }

/* ── Bouton dark-mode toggle ── */
#dark-toggle {
    flex-shrink: 0;
    background: rgba(255,255,255,0.15);
    border: 1.5px solid rgba(255,255,255,0.30);
    border-radius: 50%;
    width: 42px; height: 42px;
    font-size: 1.25em;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.2s, transform 0.2s;
    backdrop-filter: blur(4px);
    user-select: none;
}
#dark-toggle:hover { background: rgba(255,255,255,0.28); transform: scale(1.08); }

/* ── Cards ── */
.card {
    background: var(--card) !important;
    border-radius: 12px;
    padding: 20px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
    border: 1px solid var(--card-border) !important;
    transition: background 0.25s, border-color 0.25s;
}

/* ── Zone de saisie ── */
#text-input-area textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.97em !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--input-bg) !important;
    color: var(--text) !important;
    padding: 12px !important;
    transition: border-color 0.2s, background 0.25s !important;
    resize: vertical !important;
    min-height: 100px !important;
}
#text-input-area textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* ── Boutons action ── */
#btn-analyse {
    background: linear-gradient(135deg, #1d4ed8, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.97em !important;
    height: 44px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
#btn-analyse:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important; }
#btn-clear {
    border-radius: 10px !important; height: 44px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    transition: background 0.25s, border-color 0.2s !important;
}
#btn-clear:hover { filter: brightness(0.95); }
#btn-dl {
    border-radius: 10px !important; height: 44px !important;
    border: 1.5px solid #6366f1 !important;
    color: #6366f1 !important;
    background: var(--card) !important;
    font-weight: 500 !important;
}

/* ── Accordion ── */
.params-accordion > .label-wrap {
    background: var(--card) !important;
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    transition: background 0.25s;
}
.params-accordion .wrap {
    border: 1.5px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: var(--acc-body) !important;
    padding: 16px !important;
    transition: background 0.25s;
}

/* ── Tabs ── */
.tab-nav button {
    border-radius: 8px !important;
    font-size: 0.85em !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    color: var(--text-muted) !important;
}
.tab-nav button.selected {
    background: var(--tab-sel-bg) !important;
    color: var(--tab-sel-fg) !important;
    border-bottom: 2px solid var(--tab-sel-fg) !important;
}

/* ── Labels de section ── */
.section-label {
    font-size: 0.78em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-muted); margin: 14px 0 8px !important;
}

/* ── Sliders et inputs ── */
.gradio-slider input[type=range] { accent-color: #3b82f6; }
html[data-theme="dark"] label { color: var(--text) !important; }
html[data-theme="dark"] input, html[data-theme="dark"] select, html[data-theme="dark"] textarea {
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ── Zone résultat ── */
#result-card {
    background: var(--card);
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: var(--shadow-sm);
    min-height: 80px;
    line-height: 2.1;
    font-size: 1.02em;
    transition: background 0.25s;
}

/* ── Cartes de phrases (inline HTML) ── */
.sent-card {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
}
.sent-num { background: color-mix(in srgb, var(--border) 60%, transparent) !important; color: var(--text-muted) !important; }
.ner-result-wrap { color: var(--text) !important; }

/* ── Stats / détails ── */
#stats-panel, #details-panel {
    background: var(--card);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: var(--shadow-xs);
    font-size: 0.9em;
    transition: background 0.25s;
}
html[data-theme="dark"] #stats-panel *, html[data-theme="dark"] #details-panel * { color: var(--text) !important; }

/* ── CheckboxGroup ── */
.checkbox-group label {
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 4px 10px !important;
    margin: 3px !important;
    font-size: 0.85em !important;
    transition: border-color 0.15s, background 0.15s;
    color: var(--text) !important;
}
.checkbox-group label:has(input:checked) {
    border-color: var(--tab-sel-fg);
    background: var(--chk-sel-bg);
    color: var(--chk-sel-fg) !important;
}

/* ── SVG arc en dark mode ── */
html[data-theme="dark"] .svo-arc-wrap svg { filter: invert(0.88) hue-rotate(180deg) saturate(0.85); }

/* ── Examples ── */
.examples table { border-radius: 8px; overflow: hidden; }
.examples td { font-size: 0.85em !important; color: var(--text) !important; }
html[data-theme="dark"] .examples { background: var(--card) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 3px; }
"""

# ──────────────────────────────────────────────────────────
#  UI Gradio
# ──────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f0f2f5",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 2px 12px rgba(0,0,0,0.07)",
    block_radius="12px",
    input_background_fill="white",
    input_border_color="#e2e8f0",
    button_primary_background_fill="linear-gradient(135deg,#1d4ed8,#6366f1)",
    button_primary_text_color="white",
)

GLOBAL_JS = """
/* ── Helpers ── */
function _esc(s) {
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function _row(k, v) {
    return '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
}
function _code(v) { return v != null ? '<code>' + _esc(v) + '</code>' : '<span style="opacity:.4">—</span>'; }

/* ── Panneau flottant : render ── */
function _renderDetail(d) {
    if (!d) return '<p style="color:#94a3b8;margin:0">Sélectionnez un span.</p>';
    var rows = '', title = '';
    if (d.coarse !== undefined && d.fine !== undefined) {
        var lbl = (d.fine && d.fine !== 'NONE') ? d.fine : d.coarse;
        title = '🏷 <span style="font-family:monospace">' + _esc(lbl) + '</span>';
        rows += _row('Texte',    _code(d.text));
        rows += _row('Coarse',   _code(d.coarse));
        rows += _row('Fine',     _code(d.fine));
        rows += _row('Positions','[' + d.char_start + ':' + d.char_end + '] tok[' + d.tok_start + ':' + d.tok_end + ']');
        rows += '<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.85em;color:#64748b">SCORES</td></tr>';
        rows += _row('p_entity',  d.boundary_prob != null ? d.boundary_prob : '—');
        rows += _row('p_coarse',  d.coarse_prob   != null ? d.coarse_prob   : '—');
        rows += _row('p_fine',    d.fine_prob     != null ? d.fine_prob     : '—');
        rows += _row('score',     '<strong>' + (d.score != null ? d.score : '—') + '</strong>');
        if (d.ner_override) rows += _row('🔗 override', _code(d.ner_override) + ' (' + d.ner_override_score + ')');
        if (d._components && d._components.length) {
            rows += '<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.85em;color:#64748b">COMPOSANTS (' + d._components.length + ')</td></tr>';
            d._components.forEach(function(c, ci) {
                rows += '<tr><td style="padding-left:10px;color:#94a3b8;font-size:.88em">#' + (ci+1) + '</td>'
                      + '<td style="font-size:.88em"><code>' + _esc(c.fine || c.coarse) + '</code> '
                      + _code(c.text)
                      + (c.score != null ? ' <span style="opacity:.55;font-size:.9em">(' + c.score + ')</span>' : '')
                      + '</td></tr>';
            });
        }
    } else if (d.svo_role) {
        var roleEmoji = {svo_verb:'🔵',svo_subject:'🟢',svo_object:'🔴',svo_iobj:'🟠',pron_subj:'🟢',pron_obj:'🔴'};
        title = (roleEmoji[d.svo_role]||'⚪') + ' <span style="font-family:monospace">' + _esc(d.svo_role) + '</span>';
        rows += _row('Texte',     _code(d.text));
        rows += _row('Rôle',      _code(d.svo_role));
        rows += _row('Voice',     _code(d.voice));
        rows += _row('Positions', '[' + d.char_start + ':' + d.char_end + ']');
        if (d.from_ner) rows += _row('Source', '<em>synthétique NER</em>');
        if (d.ner_override) rows += _row('🔗 override', _code(d.ner_override) + ' (' + d.ner_override_score + ')');
        rows += '<tr><td colspan=2 style="padding-top:6px;font-weight:700;font-size:.85em;color:#64748b">SCORES</td></tr>';
        rows += _row('p_boundary', d.svo_boundary_prob != null ? d.svo_boundary_prob : '—');
        rows += _row('p_role',     d.svo_prob          != null ? d.svo_prob          : '—');
        rows += _row('voice conf', d.voice_prob        != null ? d.voice_prob        : '—');
        if (d.gender || d.number) rows += _row('genre/nb', _esc((d.gender||'—') + ' / ' + (d.number||'—')));
    } else {
        title = 'Données brutes';
        Object.keys(d).forEach(function(k){ rows += _row(_esc(k), _code(d[k])); });
    }
    return '<h4 style="margin:0 0 10px;padding-right:20px">' + title + '</h4>'
         + '<table>' + rows + '</table>';
}

/* ── Créer le panel une fois ── */
function _getPanel() {
    var p = document.getElementById('ner-detail-float');
    if (p) return p;
    p = document.createElement('div');
    p.id = 'ner-detail-float';
    p.innerHTML = '<button class="df-close" onclick="document.getElementById(\'ner-detail-float\').style.display=\'none\'">✕</button>'
                + '<div id="ner-detail-content"></div>';
    document.body.appendChild(p);
    /* Fermer sur clic hors panel */
    document.addEventListener('click', function(e) {
        if (!p.contains(e.target) && !e.target.closest('mark.ner-pill')) {
            p.style.display = 'none';
            document.querySelectorAll('mark.ner-pill.selected').forEach(function(m){ m.classList.remove('selected'); });
        }
    });
    return p;
}

/* ── selectSpan : clic sur un span NER → panneau flottant ── */
window.selectSpan = function(el) {
    document.querySelectorAll('mark.ner-pill.selected').forEach(function(m){ m.classList.remove('selected'); });
    el.classList.add('selected');
    var info = el.getAttribute('data-info');
    var d = null;
    try { d = JSON.parse(info); } catch(e) { console.warn('[NER] JSON parse error', e); }
    var panel = _getPanel();
    var content = document.getElementById('ner-detail-content');
    if (content) content.innerHTML = _renderDetail(d);
    panel.style.display = 'block';
    /* Positionner près du span cliqué */
    var rect = el.getBoundingClientRect();
    var py = Math.min(rect.bottom + 8, window.innerHeight - 320);
    panel.style.top  = Math.max(py, 10) + 'px';
    panel.style.right = '18px';
};

/* ── toggleDark ── */
window.toggleDark = function() {
    var html = document.documentElement;
    var isDark = html.getAttribute('data-theme') !== 'dark';
    html.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('ner-dark', isDark ? '1' : '0');
    var btn = document.getElementById('dark-toggle');
    if (btn) btn.textContent = isDark ? '☀️' : '🌙';
    var panel = document.getElementById('ner-detail-float');
    if (panel) panel.style.background = isDark ? '#1e293b' : '#fff';
};

/* ── Restaurer la préférence dark mode ── */
(function applyDark() {
    if (localStorage.getItem('ner-dark') === '1') {
        document.documentElement.setAttribute('data-theme', 'dark');
        var btn = document.getElementById('dark-toggle');
        if (btn) { btn.textContent = '☀️'; }
        else     { setTimeout(applyDark, 200); }
    }
})();
"""

with gr.Blocks(title="NER + SVO — DeBERTa", css=CSS, theme=_THEME) as demo:

    # ── Hero header ────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header-hero">
      <div id="header-hero-text">
        <h1>🔍 NER + SVO &nbsp;<span style="font-weight:400;opacity:.7;font-size:.6em">DeBERTa multitête</span></h1>
        <p>Collez du texte — analyse NER + arcs de dépendance SVO en temps réel.
           <strong style="color:rgba(255,255,255,.9)">Cliquez</strong> sur un span pour ses détails.</p>
      </div>
      <button id="dark-toggle" onclick="toggleDark()" title="Basculer dark / light mode">🌙</button>
    </div>
    """)

    # ── Saisie + boutons ───────────────────────────────────────────────────
    with gr.Group(elem_classes="card"):
        text_in = gr.Textbox(
            label="",
            placeholder="Collez votre texte ici… (paragraphe, articles, dépêches…)",
            lines=5, max_lines=24,
            elem_id="text-input-area",
            show_label=False,
        )
        with gr.Row():
            btn_analyse = gr.Button("🔍  Analyser", variant="primary", scale=4, elem_id="btn-analyse")
            btn_clear   = gr.Button("✕  Effacer",  scale=1, elem_id="btn-clear")
            dl_btn      = gr.DownloadButton("⬇  JSON", visible=False, scale=1, elem_id="btn-dl")

    # ── Accordéon paramètres ───────────────────────────────────────────────
    with gr.Accordion("⚙️  Paramètres", open=False, elem_classes="params-accordion"):
        with gr.Tabs():

            with gr.Tab("🎛 Affichage"):
                with gr.Row():
                    show_svo     = gr.Checkbox(label="Arcs SVO",                      value=True)
                    show_arcs    = gr.Checkbox(label="🏹 Visualisation displaCy",     value=True)
                    auto_split   = gr.Checkbox(label="Découpage auto en phrases",     value=True)
                    post_proc    = gr.Checkbox(label="NMS dynamique NER",             value=False)
                    do_reconcile = gr.Checkbox(label="🔗 Réconciliation NER↔SVO",     value=True)
                gr.HTML('<div class="section-label">Granularité des labels NER</div>')
                fine_for_coarse = gr.CheckboxGroup(
                    choices=_ALL_COARSE, value=_ALL_COARSE,
                    label="Afficher le label FIN pour :",
                    elem_classes="checkbox-group",
                )

            with gr.Tab("📐 Seuils inférence"):
                gr.HTML('<div class="section-label">Spans NER</div>')
                with gr.Row():
                    tau_boundary = gr.Slider(0.20, 0.90, value=0.40, step=0.05, label="tau_boundary")
                    tau_none     = gr.Slider(0.50, 1.00, value=0.99, step=0.01, label="tau_none")
                with gr.Row():
                    tau_coarse   = gr.Slider(0.00, 0.90, value=0.00, step=0.05, label="tau_coarse")
                    tau_fine     = gr.Slider(0.00, 0.90, value=0.00, step=0.05, label="tau_fine")
                gr.HTML('<div class="section-label">Spans SVO</div>')
                tau_svo = gr.Slider(0.20, 0.90, value=0.50, step=0.05, label="tau_svo boundary")

            with gr.Tab("🔗 Réconciliation"):
                with gr.Row():
                    min_ner_score_reconcile = gr.Slider(0.10, 0.95, value=0.50, step=0.05, label="min_ner (extension spans)")
                    min_ner_score_fill      = gr.Slider(0.10, 0.95, value=0.60, step=0.05, label="min_ner (sujets synthétiques)")
                max_gap_chars = gr.Slider(20, 300, value=120, step=10, label="max_gap_chars sujet synthétique")

            with gr.Tab("🏆 Priorités overlaps"):
                gr.HTML(
                    '<p style="font-size:.85em;color:#64748b;margin:0 0 8px">'
                    'Labels séparés par virgule — plus prioritaire à gauche. '
                    'Mélange fins (<code>hint_*</code>) et coarse autorisé. '
                    'À priorité égale, le span le plus long gagne.</p>'
                )
                span_priority = gr.Textbox(
                    value=DEFAULT_SPAN_PRIORITY,
                    label="Ordre de priorité (overlap NMS)",
                    lines=3,
                )

    # ── Résultat annoté ────────────────────────────────────────────────────
    result_html = gr.HTML(
        value='<div id="result-card" style="color:#94a3b8;font-style:italic">Le résultat apparaîtra ici…</div>',
        elem_id="result-card",
    )

    # ── Stats + Détails ────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        stats_out   = gr.Markdown(value="*Aucune analyse.*",   elem_id="stats-panel",   min_height=60)
        details_out = gr.Markdown(value="*Cliquez sur un span.*", elem_id="details-panel", min_height=60)

    # Champ bridge JS → Python (visible mais masqué via CSS)
    span_click_data = gr.Textbox(
        value="", visible=True, elem_id="span_click_data",
        label="span_click_data", elem_classes=["span-click-data"]
    )

    # ── Exemples ───────────────────────────────────────────────────────────
    with gr.Accordion("💡  Exemples", open=False):
        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_in, show_svo, post_proc, tau_boundary, tau_svo, auto_split],
            label="",
        )

    # ── Callbacks ──────────────────────────────────────────────────────────
    _SHARED_INPUTS = [
        text_in, show_svo, post_proc, tau_boundary, tau_svo, auto_split, do_reconcile,
        tau_none, tau_coarse, tau_fine,
        min_ner_score_reconcile, min_ner_score_fill, max_gap_chars, show_arcs,
        fine_for_coarse, span_priority,
    ]
    _SHARED_OUTPUTS = [result_html, stats_out, details_out, dl_btn]

    btn_analyse.click(fn=analyse, inputs=_SHARED_INPUTS, outputs=_SHARED_OUTPUTS)
    text_in.submit(fn=analyse,    inputs=_SHARED_INPUTS, outputs=_SHARED_OUTPUTS)
    btn_clear.click(
        fn=lambda: (
            "",
            '<div id="result-card" style="color:#94a3b8;font-style:italic">Le résultat apparaîtra ici…</div>',
            "*Aucune analyse.*", "*Cliquez sur un span.*", gr.DownloadButton(visible=False),
        ),
        outputs=[text_in, result_html, stats_out, details_out, dl_btn],
    )
    span_click_data.change(fn=format_details, inputs=[span_click_data], outputs=[details_out])


if __name__ == "__main__":
    print(f"🚀 Chargement du modèle depuis : {CHECKPOINT}")
    get_model()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, js=GLOBAL_JS)
