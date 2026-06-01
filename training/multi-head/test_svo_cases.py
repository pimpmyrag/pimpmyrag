#!/usr/bin/env python3
"""
Test du modèle v8.2/v8.3 sur cas SVO difficiles
Usage: python3 test_svo_cases.py checkpoint_best_multitask.pt
"""
import sys
import torch
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Test cases : phrases avec structures complexes
TEST_CASES = [
    {
        "id": "relative_simple",
        "text": "Le ministre que je vois parle aux journalistes",
        "expected": {
            "svo_spans": ["Le ministre", "je", "vois", "parle"],
            "verb_ptr": {
                "Le ministre": "parle",  # sujet de parle, PAS de vois
                "je": "vois",
            },
            "roles": {
                "Le ministre": "SUBJECT",
                "journalistes": "OBJECT",
            }
        }
    },
    {
        "id": "coordination",
        "text": "Le ministre a visité les travaux et a annoncé une réforme",
        "expected": {
            "verb_ptr": {
                "travaux": "a visité",  # objet de visité, pas de annoncé
                "réforme": "a annoncé",
            }
        }
    },
    {
        "id": "apposition",
        "text": "Le président, Emmanuel Macron, a déclaré une mesure importante",
        "expected": {
            "verb_ptr": {
                "Le président": "a déclaré",
                "Emmanuel Macron": "a déclaré",  # apposition
                "mesure": "a déclaré",
            }
        }
    },
    {
        "id": "verbe_support",
        "text": "Le gouvernement a pris la décision de réformer le système",
        "expected": {
            "verb_ptr": {
                "Le gouvernement": "a pris",  # sujet du verbe support
                "décision": "a pris",  # objet du verbe support, PAS de réformer
            }
        }
    },
    {
        "id": "passive",
        "text": "La réforme a été annoncée par le ministre hier soir",
        "expected": {
            "voice": "passive",
            "roles": {
                "La réforme": "SUBJECT",  # sujet grammatical (patient sémantique)
                "ministre": "OBLIQUE_AGENT",  # agent en by-phrase
            }
        }
    },
    {
        "id": "oblique_cause",
        "text": "L'accident est causé par la tempête qui frappe la région",
        "expected": {
            "verb_ptr": {
                "L'accident": "est causé",
                "tempête": "est causé",  # cause, pas sujet de frappe
                "qui": "frappe",
                "région": "frappe",
            }
        }
    },
    {
        "id": "iobj",
        "text": "Le président a remis la médaille au soldat blessé",
        "expected": {
            "roles": {
                "Le président": "SUBJECT",
                "médaille": "OBJECT",
                "soldat": "IOBJ",  # objet indirect à/de
            }
        }
    },
    {
        "id": "tcomp",
        "text": "Le ministre déclare que la situation est sous contrôle",
        "expected": {
            "verb_ptr": {
                "Le ministre": "déclare",
                "situation": "est",  # sujet du verbe de la complétive
            }
        }
    },
]

def load_model(checkpoint_path):
    """Charge le modèle depuis un checkpoint (filtrage si size mismatch)"""
    sys.path.insert(0, str(Path(__file__).parent))
    from multitask_model import SpanMultiTaskModel
    from transformers import AutoTokenizer
    import labels as L

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📦 Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    if tokenizer.model_max_length > 100000:
        tokenizer.model_max_length = 128

    print("📦 Chargement modèle...")
    # Utiliser num_coarse=10 (avec NONE) pour charger les anciens checkpoints
    model = SpanMultiTaskModel(
        model_name="microsoft/deberta-v3-base",
        num_coarse=len(L.COARSE_LABELS)  # 10 avec NONE
    ).to(device).float()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]

    # Filtrer les clés avec size mismatch (au cas où)
    model_dict = model.state_dict()
    filtered_ckpt = {}
    skipped = []
    for k, v in ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered_ckpt, strict=False)
    if skipped:
        print(f"  ⚠️  Skipped (size mismatch): {skipped}")

    model.eval()
    return model, tokenizer, device

def test_case(model, tokenizer, device, case):
    """Teste un cas et retourne les prédictions décodées"""
    from test_model_sentences_v3 import predict_text

    text = case["text"]
    ner_preds, svo_preds = predict_text(
        model, tokenizer, text, device,
        tau_boundary=0.70,
        tau_svo_boundary=0.50,
        tau_none=0.50,
        tau_coarse=0.45,
        tau_fine=0.00,
    )

    return {"ner": ner_preds, "svo": svo_preds, "text": text}


def find_span(preds, expected_text):
    """Cherche un span prédit correspondant à expected_text (exact ou partiel sans article)."""
    # Exact match d'abord
    for p in preds:
        if p['text'] == expected_text:
            return p
    # Match partiel : expected_text contenu dans le span ou span contenu dans expected_text
    for p in preds:
        if expected_text in p['text'] or p['text'] in expected_text:
            return p
    return None


def evaluate_predictions(case, predictions):
    """Compare les prédictions aux attendues"""
    print(f"\n{'='*80}")
    print(f"Test case: {case['id']}")
    print(f"{'='*80}")
    print(f"Phrase: {case['text']}")

    ner_preds = predictions["ner"]
    svo_preds = predictions["svo"]
    expected = case.get("expected", {})

    # Afficher les prédictions NER (avec role et verb_ptr si disponibles)
    print(f"\n  PRÉDICTIONS NER ({len(ner_preds)} spans):")
    for p in ner_preds[:10]:
        role_info = f" role={p['role']}" if p.get('role') else ""
        gov_info = f" gov_char={p['gov_verb_char_start']}" if p.get('gov_verb_char_start') is not None else ""
        print(f"    [{p['fine']:<20}] \"{p['text']}\" (score={p['score']:.3f}){role_info}{gov_info}")

    print(f"\n  PRÉDICTIONS SVO ({len(svo_preds)} spans):")
    for p in svo_preds[:10]:
        svo_role = p.get('svo_role', '?')
        voice = p.get('voice', '?')
        print(f"    [{svo_role:<15}] \"{p['text']}\" voice={voice:<8} prob={p.get('svo_boundary_prob', 0):.3f}")

    success = True
    text = predictions["text"]

    # Vérifier verb_ptr - cherche dans ner_preds avec matching partiel
    if "verb_ptr" in expected:
        print(f"\n  VÉRIFICATION VERB POINTER:")
        for span_text, expected_verb in expected["verb_ptr"].items():
            span = find_span(ner_preds, span_text)
            if span is not None:
                gov_char = span.get('gov_verb_char_start')
                matched_text = span['text']
                if gov_char is not None:
                    gov_tok = text[gov_char:gov_char+20].split()[0] if gov_char < len(text) else "?"
                    ok = expected_verb.startswith(gov_tok) or gov_tok in expected_verb
                    status = "✅" if ok else "⚠️ "
                    print(f"    {status} \"{span_text}\" (≈\"{matched_text}\") → attendu={expected_verb!r} prédit=@{gov_char} ({gov_tok!r})")
                    if not ok:
                        success = False
                else:
                    print(f"    ⚠️  \"{span_text}\" (≈\"{matched_text}\") → attendu={expected_verb!r} (pas de gov_verb)")
            else:
                print(f"    ❌ \"{span_text}\" → {expected_verb!r} (span NER non détecté)")
                success = False

    # Vérifier roles dans ner_preds avec matching partiel
    if "roles" in expected:
        print(f"\n  VÉRIFICATION RÔLES SVO:")
        for span_text, expected_role in expected["roles"].items():
            span = find_span(ner_preds, span_text)
            if span is not None:
                predicted_role = span.get('role', 'NONE')
                match = predicted_role == expected_role
                status = "✅" if match else "❌"
                print(f"    {status} \"{span_text}\" (≈\"{span['text']}\") attendu={expected_role:<15} prédit={predicted_role}")
                if not match:
                    success = False
            else:
                print(f"    ❌ \"{span_text}\" attendu={expected_role:<15} prédit=NOT_FOUND")
                success = False

    # Vérifier voice dans svo_preds sur les verb_trigger
    if "voice" in expected:
        verbs_with_voice = [p for p in svo_preds if p.get('voice') and p.get('svo_role') == 'verb_trigger']
        if verbs_with_voice:
            predicted_voice = verbs_with_voice[0].get('voice')
            match = predicted_voice == expected["voice"]
            status = "✅" if match else "❌"
            print(f"\n  VOICE: {status} attendu={expected['voice']} prédit={predicted_voice}")
            if not match:
                success = False
        else:
            print(f"\n  VOICE: ❌ aucun verb_trigger détecté")
            success = False

    return success
    print(f"\n{'='*80}")
    print(f"Test case: {case['id']}")
    print(f"{'='*80}")
    print(f"Phrase: {case['text']}")

    ner_preds = predictions["ner"]
    svo_preds = predictions["svo"]
    expected = case.get("expected", {})

    # Afficher les prédictions NER (avec role et verb_ptr si disponibles)
    print(f"\n  PRÉDICTIONS NER ({len(ner_preds)} spans):")
    for p in ner_preds[:10]:
        role_info = f" role={p['role']}" if p.get('role') else ""
        gov_info = f" gov_char={p['gov_verb_char_start']}" if p.get('gov_verb_char_start') is not None else ""
        print(f"    [{p['fine']:<20}] \"{p['text']}\" (score={p['score']:.3f}){role_info}{gov_info}")

    print(f"\n  PRÉDICTIONS SVO ({len(svo_preds)} spans):")
    for p in svo_preds[:10]:
        svo_role = p.get('svo_role', '?')
        voice = p.get('voice', '?')
        print(f"    [{svo_role:<15}] \"{p['text']}\" voice={voice:<8} prob={p.get('svo_boundary_prob', 0):.3f}")

    # Comparaison basique
    success = True

    # Vérifier verb_ptr si spécifié dans expected
    # On cherche dans ner_preds puisque le gouverneur est associé aux entités NER
    if "verb_ptr" in expected:
        print(f"\n  VÉRIFICATION VERB POINTER:")
        # Map : texte du span NER → char_start du verbe gouverneur
        ner_gov_map = {p['text']: p.get('gov_verb_char_start') for p in ner_preds if p.get('gov_verb_char_start') is not None}
        text = predictions["text"]

        for span_text, expected_verb in expected["verb_ptr"].items():
            gov_char = ner_gov_map.get(span_text)
            if gov_char is not None:
                # Retrouver le mot à cette position dans le texte
                gov_tok = text[gov_char:gov_char+20].split()[0] if gov_char < len(text) else "?"
                status = "✅" if expected_verb.startswith(gov_tok) or gov_tok in expected_verb else "⚠️ "
                print(f"    {status} \"{span_text}\" → attendu={expected_verb!r} prédit_pos={gov_char} ({gov_tok!r})")
            else:
                status = "❌"
                print(f"    {status} \"{span_text}\" → {expected_verb!r} (span NER non trouvé ou sans gov_verb)")
                success = False

    # Vérifier roles si spécifié (dans ner_preds)
    if "roles" in expected:
        print(f"\n  VÉRIFICATION RÔLES SVO:")
        role_map = {p['text']: p.get('role') for p in ner_preds if p.get('role')}
        for span_text, expected_role in expected["roles"].items():
            predicted_role = role_map.get(span_text, "NOT_FOUND")
            match = predicted_role == expected_role
            status = "✅" if match else "❌"
            print(f"    {status} \"{span_text}\" attendu={expected_role:<15} prédit={predicted_role}")
            if not match:
                success = False

    # Vérifier voice si spécifié (dans svo_preds, sur les verb_trigger)
    if "voice" in expected:
        verbs_with_voice = [p for p in svo_preds if p.get('voice') and p.get('svo_role') == 'verb_trigger']
        if verbs_with_voice:
            predicted_voice = verbs_with_voice[0].get('voice')
            match = predicted_voice == expected["voice"]
            status = "✅" if match else "❌"
            print(f"\n  VOICE: {status} attendu={expected['voice']} prédit={predicted_voice}")
            if not match:
                success = False
        else:
            print(f"\n  VOICE: ❌ aucun verb_trigger détecté")
            success = False

    return success

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_svo_cases.py checkpoint_best_multitask.pt")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint introuvable: {checkpoint_path}")
        sys.exit(1)

    print("📦 Chargement du modèle...")
    model, tokenizer, device = load_model(checkpoint_path)
    print(f"✅ Modèle chargé sur {device}")
    print(f"\n🧪 Test de {len(TEST_CASES)} cas SVO difficiles...\n")

    results = []
    for case in TEST_CASES:
        preds = test_case(model, tokenizer, device, case)
        success = evaluate_predictions(case, preds)
        results.append((case["id"], success))

    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")
    n_success = sum(1 for _, s in results if s)
    print(f"  {n_success}/{len(results)} tests réussis")

    for case_id, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {case_id}")

if __name__ == "__main__":
    main()

