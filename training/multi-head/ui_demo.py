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
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from test_model_sentences_v3 import load_model_and_tokenizer, predict_texts_batch, pick_device, post_process_dynamic, dedupe_overlaps

# ──────────────────────────────────────────────────────────
#  Config modèle
# ──────────────────────────────────────────────────────────

CHECKPOINT = "/Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tuning-22042026/checkpoint_best_multitask.pt"
MODEL_NAME = "microsoft/deberta-v3-base"
TOKENIZER_PATH = None  # None = utilise MODEL_NAME

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
#  Chargement modèle (lazy, une fois)
# ──────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_device = None

def get_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = pick_device()
        print(f"✅ device = {_device}")
        _model, _tokenizer = load_model_and_tokenizer(
            model_name=MODEL_NAME,
            checkpoint_path=CHECKPOINT,
            tokenizer_path=TOKENIZER_PATH,
            device=_device,
        )
        print("✅ Modèle chargé")
    return _model, _tokenizer, _device


# ──────────────────────────────────────────────────────────
#  Helpers HTML
# ──────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _span_html(text: str, bg: str, fg: str, label: str, data: dict, span_id: str) -> str:
    """Génère un <mark> cliquable avec data-info pour le panneau détails."""
    data_json = _escape(json.dumps(data, ensure_ascii=False))
    return (
        f'<mark id="{span_id}" '
        f'style="background:{bg};color:{fg};border-radius:4px;padding:1px 5px;'
        f'cursor:pointer;margin:0 1px;border:1px solid {fg}40;font-size:0.95em;" '
        f'data-info="{data_json}" '
        f'onclick="selectSpan(this)" '
        f'title="{label}">'
        f'{_escape(text)}'
        f'<sup style="font-size:0.65em;margin-left:2px;opacity:0.75">{label}</sup>'
        f'</mark>'
    )


def build_annotated_html(text: str, ner_spans: list, svo_spans: list, show_svo: bool) -> str:
    """
    Reconstruit le texte avec les spans NER et SVO surlignés.
    Les spans sont triés par position ; en cas de chevauchement NER/SVO,
    NER est prioritaire.
    """
    # Fusionner et trier (NER en priorité sur SVO si overlap)
    all_spans = []
    for i, e in enumerate(ner_spans):
        all_spans.append({
            "start": e["char_start"], "end": e["char_end"],
            "text":  e["text"],
            "kind":  "ner",
            "label": e.get("fine", e.get("coarse", "?")),
            "coarse": e.get("coarse", "?"),
            "bg":    COARSE_COLORS.get(e.get("coarse", "NONE"), COARSE_COLORS["NONE"])[0],
            "fg":    COARSE_COLORS.get(e.get("coarse", "NONE"), COARSE_COLORS["NONE"])[1],
            "data":  {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in e.items()},
            "id":    f"ner_{i}",
        })

    if show_svo:
        for i, s in enumerate(svo_spans):
            role = s.get("svo_role", "?")
            all_spans.append({
                "start":  s["char_start"], "end": s["char_end"],
                "text":   s["text"],
                "kind":   "svo",
                "label":  f'{SVO_EMOJI.get(role, "⚪")}{role}',
                "coarse": role,
                "bg":     SVO_COLORS.get(role, ("#f3f4f6", "#6b7280"))[0],
                "fg":     SVO_COLORS.get(role, ("#f3f4f6", "#6b7280"))[1],
                "data":   {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in s.items()},
                "id":     f"svo_{i}",
            })

    # Tri : par position puis NER avant SVO
    all_spans.sort(key=lambda x: (x["start"], 0 if x["kind"] == "ner" else 1))

    # Reconstruction du HTML sans overlaps
    html_parts = []
    cursor = 0
    active_end = -1

    for sp in all_spans:
        s, e = sp["start"], sp["end"]
        if s < active_end:
            continue  # skip overlapping span
        if s > cursor:
            html_parts.append(_escape(text[cursor:s]))
        html_parts.append(_span_html(sp["text"], sp["bg"], sp["fg"],
                                      sp["label"], sp["data"], sp["id"]))
        cursor = e
        active_end = e

    if cursor < len(text):
        html_parts.append(_escape(text[cursor:]))

    return "".join(html_parts)


def build_legend_html() -> str:
    items = []
    for coarse, (bg, fg) in COARSE_COLORS.items():
        if coarse == "NONE":
            continue
        items.append(
            f'<span style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:2px 7px;margin:2px;font-size:0.82em;border:1px solid {fg}40">'
            f'{coarse}</span>'
        )
    items.append('<span style="margin-left:12px;font-weight:600;font-size:0.82em">SVO : </span>')
    for role, (bg, fg) in SVO_COLORS.items():
        emoji = SVO_EMOJI.get(role, "")
        items.append(
            f'<span style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:2px 7px;margin:2px;font-size:0.82em;border:1px solid {fg}40">'
            f'{emoji}{role}</span>'
        )
    return '<div style="margin:8px 0;line-height:2">' + "".join(items) + "</div>"

# ──────────────────────────────────────────────────────────
#  JS intégré — handleClick → met à jour le champ hidden
# ──────────────────────────────────────────────────────────

JS_CLICK = """
<script>
function selectSpan(el) {
    // Retirer la sélection précédente
    document.querySelectorAll('mark.selected-span').forEach(m => {
        m.classList.remove('selected-span');
        m.style.outline = '';
    });
    el.classList.add('selected-span');
    el.style.outline = '3px solid #f59e0b';

    const info = el.getAttribute('data-info');
    // Trouver la textarea cachée de Gradio et y injecter la valeur
    const hiddenBox = document.querySelector('#span_click_data textarea');
    if (hiddenBox) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value').set;
        nativeInputValueSetter.call(hiddenBox, info);
        hiddenBox.dispatchEvent(new Event('input', { bubbles: true }));
    }
}
</script>
<style>
mark { transition: outline 0.15s; }
mark:hover { filter: brightness(0.93); }
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
        lines.append(f"**Positions** : char [{d.get('char_start')} : {d.get('char_end')}]  |  tok [{d.get('tok_start')} : {d.get('tok_end')}]")
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
#  Fonction analyse principale
# ──────────────────────────────────────────────────────────

def analyse(text_input: str, show_svo: bool, post_process: bool,
            tau_boundary: float, tau_svo: float) -> tuple:
    if not text_input.strip():
        return "<i>Entrez du texte ci-dessus.</i>", "", "*Aucun résultat.*"

    model, tokenizer, device = get_model()

    # Découper en phrases (une par ligne, ou phrase unique)
    phrases = [l.strip() for l in text_input.strip().splitlines() if l.strip()]

    results = predict_texts_batch(
        model=model, tokenizer=tokenizer, texts=phrases, device=device,
        max_length=128, max_span_len=12,
        tau_boundary=tau_boundary, tau_none=0.99, tau_coarse=0.00,
        tau_fine=0.00, topk_coarse=2, min_char_len=2,
        enforce_word_boundaries=True, tau_svo_boundary=tau_svo,
    )

    html_blocks = []
    stats_lines = []
    total_ner = 0
    total_svo = 0

    for i, (phrase, res) in enumerate(zip(phrases, results)):
        ner = res["ner"]
        svo = res["svo"]
        if post_process:
            ner = post_process_dynamic(ner)
        else:
            ner = dedupe_overlaps(ner)

        total_ner += len(ner)
        total_svo += len(svo)

        block = build_annotated_html(phrase, ner, svo if show_svo else [], show_svo)

        if len(phrases) > 1:
            html_blocks.append(
                f'<div style="margin-bottom:18px">'
                f'<span style="font-size:0.75em;color:#64748b;font-weight:600">#{i+1}</span> '
                f'{block}</div>'
            )
        else:
            html_blocks.append(f'<div style="line-height:2.0;font-size:1.05em">{block}</div>')

        # Stats par phrase
        coarse_counts = {}
        for e in ner:
            c = e.get("coarse", "?")
            coarse_counts[c] = coarse_counts.get(c, 0) + 1
        svo_counts = {}
        for s in svo:
            r = s.get("svo_role", "?")
            svo_counts[r] = svo_counts.get(r, 0) + 1

        stats_lines.append(f"**#{i+1}** — NER: {len(ner)} | SVO: {len(svo)}")
        for c, n in sorted(coarse_counts.items()):
            stats_lines.append(f"  - {c}: {n}")
        for r, n in sorted(svo_counts.items()):
            stats_lines.append(f"  - {SVO_EMOJI.get(r,'')} {r}: {n}")

    legend = build_legend_html()
    main_html = (
        JS_CLICK
        + legend
        + '<div style="font-family:system-ui,sans-serif;padding:4px 0">'
        + "\n".join(html_blocks)
        + "</div>"
    )

    stats_md = f"**{len(phrases)} phrase(s)** — **NER total : {total_ner}** | **SVO total : {total_svo}**\n\n"
    stats_md += "\n".join(stats_lines)

    return main_html, stats_md, ""  # 3e sortie = détails (réinitialisé)


# ──────────────────────────────────────────────────────────
#  UI Gradio
# ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["Emmanuel Macron s'est rendu à Berlin pour rencontrer Olaf Scholz.", True, False, 0.40, 0.50],
    ["La Banque centrale européenne a relevé ses taux d'intérêt de 25 points de base mardi.", True, False, 0.40, 0.50],
    ["Apple a annoncé le lancement de l'iPhone 17 le 15 septembre 2025 à Cupertino.\nTesla a livré 500 000 véhicules au troisième trimestre.", True, False, 0.40, 0.50],
    ["Le Conseil constitutionnel a censuré plusieurs articles de la loi immigration.\nLa directive européenne sur l'IA entre en vigueur le 1er août.", True, True, 0.40, 0.50],
]

with gr.Blocks(title="NER + SVO Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔍 NER + SVO — DeBERTa multitête\n"
        "Collez du texte (une ou plusieurs phrases, une par ligne). "
        "**Cliquez** sur un span surligné pour voir ses scores complets."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(
                label="Texte à analyser",
                placeholder="Collez votre texte ici… (une phrase par ligne pour le mode batch)",
                lines=5,
                max_lines=20,
            )
            with gr.Row():
                btn_analyse = gr.Button("🔍 Analyser", variant="primary", scale=2)
                btn_clear   = gr.Button("🗑 Effacer", scale=1)

            with gr.Accordion("⚙️ Paramètres", open=False):
                show_svo     = gr.Checkbox(label="Afficher les spans SVO (verbes, sujets, objets)", value=True)
                post_proc    = gr.Checkbox(label="Post-processing NMS dynamique (NER)", value=False)
                tau_boundary = gr.Slider(0.2, 0.9, value=0.40, step=0.05, label="Seuil boundary NER (tau_boundary)")
                tau_svo      = gr.Slider(0.2, 0.9, value=0.50, step=0.05, label="Seuil SVO boundary (tau_svo)")

        with gr.Column(scale=1):
            stats_out   = gr.Markdown(label="Statistiques", value="*Aucune analyse.*")
            details_out = gr.Markdown(label="Détail du span sélectionné",
                                       value="*Cliquez sur un span surligné.*")

    # Zone principale de résultat
    result_html = gr.HTML(label="Texte annoté", value="<i>Le résultat apparaîtra ici.</i>")

    # Champ caché pour le bridge JS → Python (clic sur span)
    span_click_data = gr.Textbox(
        value="", visible=False, elem_id="span_click_data", label="span_click_data"
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_in, show_svo, post_proc, tau_boundary, tau_svo],
        label="Exemples",
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    btn_analyse.click(
        fn=analyse,
        inputs=[text_in, show_svo, post_proc, tau_boundary, tau_svo],
        outputs=[result_html, stats_out, details_out],
    )
    text_in.submit(
        fn=analyse,
        inputs=[text_in, show_svo, post_proc, tau_boundary, tau_svo],
        outputs=[result_html, stats_out, details_out],
    )
    btn_clear.click(
        fn=lambda: ("", "<i>Le résultat apparaîtra ici.</i>", "*Aucune analyse.*", "*Cliquez sur un span.*"),
        outputs=[text_in, result_html, stats_out, details_out],
    )

    # Clic sur span → détails (via champ hidden mis à jour par JS)
    span_click_data.change(
        fn=format_details,
        inputs=[span_click_data],
        outputs=[details_out],
    )


if __name__ == "__main__":
    print(f"🚀 Chargement du modèle depuis : {CHECKPOINT}")
    get_model()  # pré-charger au démarrage
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

