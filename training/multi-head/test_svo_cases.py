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
    """Charge le modèle depuis un checkpoint"""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_model_sentences_v3 import load_model_and_tokenizer
    import labels as L

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(
        model_name="microsoft/deberta-v3-base",
        checkpoint_path=str(checkpoint_path),
        tokenizer_path="microsoft/deberta-v3-base",
        device=device
    )
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


def evaluate_predictions(case, predictions):
    """Compare les prédictions aux attendues"""
    print(f"\n{'='*80}")
    print(f"Test case: {case['id']}")
    print(f"{'='*80}")
    print(f"Phrase: {case['text']}")

    ner_preds = predictions["ner"]
    svo_preds = predictions["svo"]
    expected = case.get("expected", {})

    # Afficher les prédictions
    print(f"\n  PRÉDICTIONS NER ({len(ner_preds)} spans):")
    for p in ner_preds[:10]:
        print(f"    [{p['fine']:<20}] \"{p['text']}\" (score={p['score']:.3f})")

    print(f"\n  PRÉDICTIONS SVO ({len(svo_preds)} spans):")
    for p in svo_preds[:10]:
        syn = p.get('syn', '?')
        role = p.get('role', '?')
        gov = p.get('gov_verb_char_start', '?')
        print(f"    [{syn:<15}] \"{p['text']}\" role={role:<15} gov_verb_pos={gov}")

    # Comparaison basique
    success = True

    # Vérifier verb_ptr si spécifié dans expected
    if "verb_ptr" in expected:
        print(f"\n  VÉRIFICATION VERB POINTER:")
        verb_ptr_map = defaultdict(list)
        for p in svo_preds:
            if p.get('gov_verb_char_start') is not None:
                verb_ptr_map[p['text']].append(p.get('gov_verb_char_start'))

        for span_text, expected_verb in expected["verb_ptr"].items():
            found = span_text in verb_ptr_map
            status = "✅" if found else "❌"
            print(f"    {status} \"{span_text}\" → {expected_verb}")
            if not found:
                success = False

    # Vérifier roles si spécifié
    if "roles" in expected:
        print(f"\n  VÉRIFICATION RÔLES SVO:")
        role_map = {p['text']: p.get('role') for p in svo_preds if p.get('role')}
        for span_text, expected_role in expected["roles"].items():
            predicted_role = role_map.get(span_text, "NOT_FOUND")
            match = predicted_role == expected_role
            status = "✅" if match else "❌"
            print(f"    {status} \"{span_text}\" attendu={expected_role:<15} prédit={predicted_role}")
            if not match:
                success = False

    # Vérifier voice si spécifié
    if "voice" in expected:
        verbs_with_voice = [p for p in svo_preds if p.get('voice')]
        if verbs_with_voice:
            predicted_voice = verbs_with_voice[0].get('voice')
            match = predicted_voice == expected["voice"]
            status = "✅" if match else "❌"
            print(f"\n  VOICE: {status} attendu={expected['voice']} prédit={predicted_voice}")
            if not match:
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

