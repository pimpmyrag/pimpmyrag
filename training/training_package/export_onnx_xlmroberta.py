"""
Export + quantification INT8 XLMRobertaForTokenClassification via Hugging Face Optimum.

Optimum gère nativement l'opset, les axes dynamiques et la quantification INT8 compatible ORT,
sans produire de DynamicQuantizeMatMul avec contrainte d'alignement seq_len % 8.

Usage :
    # Export FP32 + quantification INT8 (défaut)
    python export_onnx_xlmroberta.py \
        --model_dir training_output/checkpoint-7800 \
        --output    training_output/model_v5.onnx

    # Export FP32 seulement
    python export_onnx_xlmroberta.py --model_dir ... --output ... --no-quantize
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import onnx
from transformers import AutoTokenizer


def export(model_dir: str, output_path: str, quantize: bool = True):
    from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    output_path = Path(output_path)
    output_dir  = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Export ONNX via optimum ──────────────────────────────────────────
    print(f"[1/4] Export ONNX depuis : {model_dir}")
    with tempfile.TemporaryDirectory(prefix="optimum_export_") as tmp_dir:
        ort_model = ORTModelForTokenClassification.from_pretrained(
            model_dir,
            export=True,
            provider="CPUExecutionProvider",
        )
        num_labels = ort_model.config.num_labels
        print(f"      num_labels = {num_labels}")

        # Sauvegarde dans un répertoire temporaire (optimum sauve model.onnx + tokenizer)
        ort_model.save_pretrained(tmp_dir)
        AutoTokenizer.from_pretrained(model_dir).save_pretrained(tmp_dir)

        tmp_onnx = Path(tmp_dir) / "model.onnx"

        # ── 2. Validation FP32 ─────────────────────────────────────────────
        print("[2/4] Validation du modèle FP32…")
        onnx.checker.check_model(str(tmp_onnx))
        print("      ✅ FP32 valide.")

        if not quantize:
            # Copie du FP32 vers la destination finale
            _copy_onnx(tmp_onnx, output_path)
            print(f"      Fichier FP32 écrit : {output_path}")
            _print_labels(ort_model)
            return

        # ── 3. Quantification INT8 via ORTQuantizer ────────────────────────
        print("[3/4] Quantification INT8…")
        # arm64 = Apple Silicon ; pour serveur Intel utiliser avx2 ou avx512_vnni
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

        quantizer = ORTQuantizer.from_pretrained(tmp_dir)
        with tempfile.TemporaryDirectory(prefix="optimum_quant_") as q_dir:
            quantizer.quantize(save_dir=q_dir, quantization_config=qconfig)

            q_onnx = Path(q_dir) / "model_quantized.onnx"
            if not q_onnx.exists():
                # Certaines versions d'optimum nomment autrement
                candidates = list(Path(q_dir).glob("*.onnx"))
                if not candidates:
                    raise FileNotFoundError(f"Aucun .onnx trouvé dans {q_dir}")
                q_onnx = candidates[0]

            print(f"      Fichier INT8 brut : {q_onnx.name}")

            # ── 4. Validation INT8 + copie finale ─────────────────────────
            print("[4/4] Validation du modèle INT8…")
            import onnxruntime as ort
            sess = ort.InferenceSession(str(q_onnx), providers=["CPUExecutionProvider"])
            print(f"      ✅ INT8 valide — inputs : {[i.name for i in sess.get_inputs()]}")
            del sess

            _copy_onnx(q_onnx, output_path)

        fp32_path = output_dir / (output_path.stem + "_fp32.onnx")
        _copy_onnx(tmp_onnx, fp32_path)
        print(f"\n      FP32 sauvegardé   : {fp32_path}")
        print(f"      INT8 sauvegardé   : {output_path}")

    _print_labels(ort_model)


def _copy_onnx(src: Path, dst: Path):
    """Copie src (+ éventuel .onnx.data) vers dst."""
    shutil.copy2(src, dst)
    src_data = src.parent / (src.name + ".data")
    if src_data.exists():
        shutil.copy2(src_data, dst.parent / (dst.name + ".data"))


def _print_labels(ort_model):
    print("\nLabels :")
    for idx, label in ort_model.config.id2label.items():
        print(f"  {str(idx):>3} -> {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export XLM-RoBERTa NER → ONNX INT8 via optimum")
    parser.add_argument("--model_dir", default="training_output/checkpoint-7800",
                        help="Répertoire du checkpoint HuggingFace")
    parser.add_argument("--output",    default="training_output/model_v5.onnx",
                        help="Chemin de sortie (INT8 si --quantize, FP32 sinon)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Export FP32 seulement, sans quantification INT8")
    args = parser.parse_args()

    export(
        model_dir=args.model_dir,
        output_path=args.output,
        quantize=not args.no_quantize,
    )
