#!/usr/bin/env python3
"""
quantize_onnx.py
~~~~~~~~~~~~~~~~
Quantification dynamique int8 du modèle ONNX exporté.

La quantification dynamique :
  - Stocke les poids des MatMul en int8  (4× moins de mémoire = 4× moins de cache pollution)
  - Calcule les activations en int8 à la volée
  - Typiquement  2-4× plus rapide sur CPU sans perte significative de précision (<1% F1)

Usage :
    python quantize_onnx.py --input best_model_multitask_full.onnx \
                             --output best_model_multitask_q8.onnx
"""
import argparse
import os
from pathlib import Path


def quantize(input_path: str, output_path: str, per_channel: bool = False) -> None:
    """
    Quantification dynamique int8.

    per_channel=False  → quantification par tenseur → PLUS RAPIDE sur Apple Silicon ARM
    per_channel=True   → plus précis mais plus lent à déquantifier sur ARM
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"⚙️  Quantification int8 (per_channel={per_channel})")
    print(f"   Input  : {input_path}  ({os.path.getsize(input_path)/1e6:.1f} Mo)")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,   # True uniquement pour Intel VNNI ; sur ARM = inutile
        optimize_model=True,  # lance une passe d'optimisation ORT avant quant.
    )

    size_in  = os.path.getsize(input_path)  / 1e6
    size_out = os.path.getsize(output_path) / 1e6
    print(f"✅ Terminé : {output_path}  ({size_out:.1f} Mo  — {size_in/size_out:.1f}× plus petit)")


def verify(model_path: str) -> None:
    """Vérification rapide : inférence dummy avec le modèle quantifié."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    B, L, N = 1, 32, 10
    feeds = {
        "input_ids":      np.zeros((B, L), dtype=np.int64),
        "attention_mask": np.ones  ((B, L), dtype=np.int64),
        "span_starts":    np.arange(N,      dtype=np.int64) % (L - 4) + 2,
        "span_ends":      (np.arange(N,     dtype=np.int64) % (L - 4) + 4).clip(max=L-2),
        "span_batch_ids": np.zeros(N,       dtype=np.int64),
    }
    outs = sess.run(None, feeds)
    shapes = "  ".join(f"{o.shape}" for o in outs)
    print(f"✅ Vérification OK — sorties : {shapes}")


def main():
    ap = argparse.ArgumentParser(description="Quantification dynamique int8 du modèle ONNX")
    ap.add_argument("--input",       required=True,        help="Modèle ONNX source (float32)")
    ap.add_argument("--output",      required=True,        help="Modèle ONNX quantifié (int8)")
    ap.add_argument("--per-channel", action="store_true",  help="Quant. par canal (plus précis, plus lent sur ARM)")
    ap.add_argument("--no-verify",   action="store_true",  help="Sauter la vérification post-quant")
    args = ap.parse_args()

    quantize(args.input, args.output, per_channel=args.per_channel)
    if not args.no_verify:
        verify(args.output)


if __name__ == "__main__":
    main()

