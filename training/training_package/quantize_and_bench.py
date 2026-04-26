#!/usr/bin/env python3
"""
Quantifie en INT8 (dynamic) les modèles ONNX référencés dans application.yml
et compare les performances (taille, latence, fidélité des prédictions).

Deux modèles sont traités :
  • BILOU XLM-RoBERTa  (onnx.ner.label) — inputs: input_ids, attention_mask
  • SpanNER DeBERTa-v3 (onnx.ner.ud)    — inputs: + span_starts/ends/batch_idx/coarse_ids

Le modèle embedding est déjà quantifié (model_quantized.onnx) → ignoré.

Usage :
  # Lecture automatique des chemins depuis application.yml
  python quantize_and_bench.py \\
      --app-yml ../../radar-nli-toolkit/src/main/resources/application.yml

  # Chemins explicites
  python quantize_and_bench.py \\
      --bilou-model training_output/model_v2.onnx \\
      --span-model  best_model_v3.onnx

  # Quantifier seulement, sans benchmark (plus rapide)
  python quantize_and_bench.py --app-yml ... --no-bench

  # Réglages fins
  python quantize_and_bench.py --app-yml ... \\
      --op-types MatMul Gather \\
      --per-channel          \\   # meilleure qualité, légèrement plus lent
      --reduce-range         \\   # pour CPU sans AVX-512 (vieux Intel)
      --warmup 10 --runs 100
"""

import os
import sys
import time
import argparse
import textwrap
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Callable

import numpy as np

# ── Dépendances optionnelles ────────────────────────────────────────────────
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    _HAS_ORT = True
except ImportError:
    print("❌  onnxruntime non disponible.  pip install onnxruntime")
    sys.exit(1)


# ============================================================================
# Helpers — fichiers
# ============================================================================

def file_size_mb(path: str) -> float:
    """Taille totale (modèle + éventuel .data extern) en Mo."""
    total = Path(path).stat().st_size
    data = Path(path.replace(".onnx", ".onnx.data"))
    if data.exists():
        total += data.stat().st_size
    return total / 1_000_000


def output_path(src: str) -> str:
    """model.onnx → model_int8.onnx dans le même répertoire."""
    p = Path(src)
    return str(p.parent / (p.stem + "_int8" + p.suffix))


# ============================================================================
# Quantification
# ============================================================================

def uses_external_data(model_path: str) -> bool:
    """Vrai si le modèle possède un fichier .onnx.data à côté."""
    return Path(model_path.replace(".onnx", ".onnx.data")).exists()


def _run_quantize_dynamic(src: str, dst: str, op_types: list, per_channel: bool,
                           reduce_range: bool, ext_data: bool) -> None:
    """Appel direct à quantize_dynamic. Lève une exception si ça échoue."""
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        op_types_to_quantize=op_types,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=[],
        use_external_data_format=ext_data,
        extra_options={"MatMulConstBOnly": True},
    )


def _find_fp16_downstream(model) -> set:
    """
    Retourne les noms de TOUS les tenseurs FP16, par propagation en point fixe :
    - initializers FP16
    - sorties de nœuds Cast→FP16
    - sorties de n'importe quel nœud dont AU MOINS UNE entrée est FP16
      (Mul, Add, Reshape, Concat, etc. — DeBERTa-v3 propage du FP16 partout)
    """
    from onnx import TensorProto as _TP
    fp16 = set()
    # Graine : initializers FP16
    for init in model.graph.initializer:
        if init.data_type == _TP.FLOAT16:
            fp16.add(init.name)
    # Graine : Cast explicites → FP16
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == _TP.FLOAT16:
                    fp16.update(node.output)
    # Point fixe : tout nœud dont une entrée est FP16 propage FP16 en sortie
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if any(inp in fp16 for inp in node.input if inp):
                for out in node.output:
                    if out and out not in fp16:
                        fp16.add(out)
                        changed = True
    return fp16


def _matmul_nodes_to_exclude(model, fp16_tensors: set) -> list:
    """
    Retourne les noms des nœuds MatMul/Gemm dont AU MOINS UNE entrée est FP16.
    Tous les nœuds doivent avoir été nommés au préalable (via _ensure_node_names).
    """
    exclude = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            if any(inp in fp16_tensors for inp in node.input if inp):
                exclude.append(node.name)  # toujours non-vide après _ensure_node_names
    return exclude


def _cast_fp16_initializers_to_fp32(model) -> int:
    """
    Convertit tous les initialiseurs FP16 → FP32 dans le graphe ONNX en mémoire.
    Cela élimine les DynamicQuantizeLinear FP16 invalides (cat_1, mul_16…) sans
    modifier l'architecture (les nœuds Cast→FP16 downstream restent valides).
    Retourne le nombre d'initialiseurs convertis.
    """
    import numpy as np
    from onnx import numpy_helper, TensorProto as _TP

    count = 0
    for init in model.graph.initializer:
        if init.data_type == _TP.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            count += 1
    # Constant nodes FP16 → FP32
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and hasattr(attr, 't') and attr.t.data_type == _TP.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
                    count += 1
    return count


def _ensure_node_names(model) -> int:
    """Donne un nom unique à tout nœud qui n'en a pas. Retourne le nombre de nœuds renommés."""
    renamed = 0
    for i, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f"_autoname_{node.op_type}_{i}"
            renamed += 1
    return renamed


def _run_quantize_no_shape_infer(src: str, dst: str, op_types: list,
                                  per_channel: bool, reduce_range: bool,
                                  ext_data: bool) -> None:
    """
    Stratégie robuste pour les modèles DeBERTa-v3 mixed-precision :
    1. Charge le modèle SANS modifier les poids FP16 (l'architecture les utilise
       intentionnellement — les changer casserait les prédictions)
    2. Détecte les nœuds MatMul dont l'activation (input[0]) est FP16
       et les ajoute à nodes_to_exclude pour éviter DynamicQuantizeLinear FP16
    3. Supprime les value_info conflictuels (→ ShapeInferenceError)
    4. Sauvegarde dans un répertoire temporaire et quantifie
    """
    import onnx as _onnx

    tmp_dir = tempfile.mkdtemp(prefix="ort_stripped_")
    try:
        print(f"         Chargement du modèle en mémoire…")
        model = _onnx.load(src)


        # ── Nommage des nœuds anonymes (requis pour nodes_to_exclude) ────────
        renamed = _ensure_node_names(model)
        if renamed:
            print(f"         {renamed} nœuds anonymes renommés")

        # ── Détection des MatMul avec activation FP16 → à exclure ──────────
        fp16_tensors  = _find_fp16_downstream(model)
        nodes_excl    = _matmul_nodes_to_exclude(model, fp16_tensors)
        print(f"         {len(fp16_tensors)} tenseurs FP16 détectés, "
              f"{len(nodes_excl)} MatMul exclus (activation FP16)")

        # ── Suppression des value_info conflictuels ──────────────────────────
        n_vi = len(model.graph.value_info)
        del model.graph.value_info[:]
        print(f"         {n_vi} value_info supprimés")

        stripped_path = str(Path(tmp_dir) / "stripped.onnx")
        if ext_data:
            _onnx.save(model, stripped_path,
                       save_as_external_data=True,
                       all_tensors_to_one_file=True,
                       location="stripped.onnx.data")
        else:
            _onnx.save(model, stripped_path)
        del model

        quantize_dynamic(
            model_input=stripped_path,
            model_output=dst,
            op_types_to_quantize=op_types,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=nodes_excl,
            use_external_data_format=ext_data,
            extra_options={"MatMulConstBOnly": True},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def quantize_model(
    src:          str,
    dst:          str,
    op_types:     list[str],
    per_channel:  bool,
    reduce_range: bool,
) -> bool:
    """
    Applique quantize_dynamic INT8 sur src → dst.

    Stratégie en cascade :
    1. Appel direct                        → le plus rapide
    2. quant_pre_process(skip_onnx_shape)  → si ShapeInferenceError de pre-process
    3. Monkey-patch load_model_with_shape_infer → contourne l'inférence ONNX dans
       quantize_dynamic lui-même (utile pour DeBERTa-v3 / custom ops PyTorch)
    """
    ext_data = uses_external_data(src)
    print(f"\n  ⚙️   Quantification {Path(src).name} → {Path(dst).name}")
    if ext_data:
        print(f"  ℹ️   Données externes détectées — use_external_data_format=True")

    # ── Tentative 1 : quantification directe ─────────────────────────────────
    try:
        _run_quantize_dynamic(src, dst, op_types, per_channel, reduce_range, ext_data)
        print(f"  ✅  Quantification OK")
        return True
    except Exception as e:
        err = str(e)
        if "ShapeInferenceError" not in err:
            print(f"  ❌  Quantification échouée : {e}")
            return False
        print(f"  ⚠️   ShapeInferenceError ({err[:80]})")

    # ── Tentative 2 : quant_pre_process skip_onnx_shape ──────────────────────
    print(f"  🔄  Tentative 2 : pre-process avec skip_onnx_shape=True…")
    try:
        from onnxruntime.quantization import quant_pre_process
        tmp_dir = tempfile.mkdtemp(prefix="ort_preproc_")
        try:
            pre_path = str(Path(tmp_dir) / "preprocessed.onnx")
            quant_pre_process(
                input_model=src,
                output_model_path=pre_path,
                skip_optimization=False,
                skip_onnx_shape=True,
                skip_symbolic_shape=True,
                save_as_external_data=ext_data,
                all_tensors_to_one_file=True,
            )
            _run_quantize_dynamic(pre_path, dst, op_types, per_channel, reduce_range, ext_data)
            print(f"  ✅  Quantification OK (pre-process skip_onnx_shape)")
            return True
        except Exception as e2:
            if "ShapeInferenceError" not in str(e2):
                print(f"  ❌  Tentative 2 échouée : {e2}")
                return False
            print(f"  ⚠️   ShapeInferenceError persistante après pre-process")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except ImportError:
        pass

    # ── Tentative 3 : monkey-patch load_model_with_shape_infer ───────────────
    print(f"  🔄  Tentative 3 : bypass shape inference via monkey-patch…")
    try:
        _run_quantize_no_shape_infer(src, dst, op_types, per_channel, reduce_range, ext_data)
        print(f"  ✅  Quantification OK (bypass shape inference)")
        return True
    except Exception as e3:
        print(f"  ❌  Toutes les tentatives ont échoué : {e3}")
        return False


# ============================================================================
# Sessions ONNX Runtime
# ============================================================================

def make_session(model_path: str, threads: int = 4) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def session_input_names(session: ort.InferenceSession) -> set[str]:
    return {inp.name for inp in session.get_inputs()}


# ============================================================================
# Benchmark
# ============================================================================

def run_bench(
    session: ort.InferenceSession,
    feed:    dict,
    warmup:  int,
    runs:    int,
) -> tuple[float, float, float]:
    """Retourne (mean_ms, min_ms, max_ms)."""
    names = session_input_names(session)
    f = {k: v for k, v in feed.items() if k in names}
    for _ in range(warmup):
        session.run(None, f)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, f)
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return float(a.mean()), float(a.min()), float(a.max())


def compare_outputs(
    sess_fp: ort.InferenceSession,
    sess_i8: ort.InferenceSession,
    feed:    dict,
) -> list[dict]:
    """
    Passe le même feed dans les deux sessions et compare les logits.
    Retourne une liste de dicts par output : {mae, top1_match, p99_err}.
    """
    names_fp = session_input_names(sess_fp)
    names_i8 = session_input_names(sess_i8)
    out_fp = sess_fp.run(None, {k: v for k, v in feed.items() if k in names_fp})
    out_i8 = sess_i8.run(None, {k: v for k, v in feed.items() if k in names_i8})
    stats = []
    for fp, i8 in zip(out_fp, out_i8):
        fp, i8 = np.asarray(fp, dtype=np.float32), np.asarray(i8, dtype=np.float32)
        diff = np.abs(fp - i8)
        mae  = float(diff.mean())
        p99  = float(np.percentile(diff, 99))
        top1 = None
        if fp.ndim >= 2:
            top1 = float((np.argmax(fp, axis=-1) == np.argmax(i8, axis=-1)).mean()) * 100
        stats.append({"mae": mae, "p99": p99, "top1_match_pct": top1})
    return stats


# ============================================================================
# Inputs factices
# ============================================================================

def bilou_inputs(seq_len: int = 128, batch: int = 1) -> dict:
    """Inputs pour XLM-RoBERTa BILOU (token classification)."""
    return {
        "input_ids":      np.random.randint(1, 32000, (batch, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
        # token_type_ids facultatif — filtré si absent du modèle
        "token_type_ids": np.zeros((batch, seq_len), dtype=np.int64),
    }


def span_inputs(seq_len: int = 64, batch: int = 2, n_spans: int = 8) -> dict:
    """Inputs pour SpanNER DeBERTa-v3 (span classification)."""
    assert n_spans % batch == 0, "n_spans doit être divisible par batch"
    per_doc = n_spans // batch
    span_batch_idx = np.repeat(np.arange(batch, dtype=np.int64), per_doc)
    span_starts    = np.random.randint(1, seq_len // 2, n_spans, dtype=np.int64)
    span_ends      = np.clip(
        span_starts + np.random.randint(1, 6, n_spans).astype(np.int64),
        a_min=span_starts + 1,
        a_max=seq_len - 1,
    )
    coarse_ids = np.random.randint(0, 6, n_spans, dtype=np.int64)
    return {
        "input_ids":      np.random.randint(1, 128000, (batch, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
        # token_type_ids facultatif — filtré si absent du modèle
        "token_type_ids": np.zeros((batch, seq_len), dtype=np.int64),
        "span_starts":    span_starts,
        "span_ends":      span_ends,
        "span_batch_idx": span_batch_idx,
        "coarse_ids":     coarse_ids,
    }


# ============================================================================
# Pipeline complet par modèle
# ============================================================================

def process_model(
    name:         str,
    src:          str,
    input_fn:     Callable[[], dict],
    op_types:     list[str],
    per_channel:  bool,
    reduce_range: bool,
    warmup:       int,
    runs:         int,
    no_bench:     bool,
) -> None:
    bar = "─" * 70
    print(f"\n╔{bar}╗")
    print(f"║  {name:<68}║")
    print(f"╚{bar}╝")
    print(f"  Source : {src}")

    if not Path(src).exists():
        print("  ❌  Fichier introuvable — skip.")
        return

    src_mb = file_size_mb(src)
    dst    = output_path(src)
    print(f"  Taille FP32 : {src_mb:.1f} MB")

    # ── Quantification ───────────────────────────────────────────────────────
    ok = quantize_model(src, dst, op_types, per_channel, reduce_range)
    if not ok:
        return

    dst_mb = file_size_mb(dst)

    # ── Validation : le modèle INT8 est-il chargeable ? ───────────────────────
    try:
        _s = make_session(dst, threads=1)
        del _s
    except Exception as load_err:
        err_s = str(load_err)
        # Cas connu : Gather avec weights FP16 → DequantizeLinear scale float16 invalide
        if ("DequantizeLinear" in err_s or "DynamicQuantizeLinear" in err_s) and ("float16" in err_s or "float 16" in err_s) and "Gather" in op_types:
            print(f"  ⚠️   Gather FP16 invalide dans le modèle INT8 — re-quantification sans Gather…")
            op_types_retry = [o for o in op_types if o != "Gather"]
            ok = quantize_model(src, dst, op_types_retry, per_channel, reduce_range)
            if not ok:
                return
            dst_mb = file_size_mb(dst)
            # Re-valide
            try:
                _s = make_session(dst, threads=1); del _s
            except Exception as e2:
                print(f"  ❌  Modèle INT8 invalide même sans Gather : {e2}")
                return
        else:
            print(f"  ❌  Modèle INT8 invalide : {load_err}")
            return
    ratio  = src_mb / dst_mb if dst_mb > 0 else float("inf")
    print(f"  Taille INT8 : {dst_mb:.1f} MB  →  ratio {ratio:.2f}×  "
          f"({(1 - dst_mb / src_mb) * 100:.0f}% plus petit)")
    print(f"  Sortie      : {dst}")

    if no_bench:
        return

    # ── Chargement des sessions ───────────────────────────────────────────────
    print(f"\n  Chargement des sessions ONNX Runtime…")
    try:
        sess_fp = make_session(src)
        sess_i8 = make_session(dst)
    except Exception as e:
        print(f"  ❌  Impossible de charger une session : {e}")
        return

    feed = input_fn()

    # ── Benchmark latence ────────────────────────────────────────────────────
    print(f"  Benchmark ({warmup} warmup + {runs} runs)…")
    try:
        mean_fp, min_fp, max_fp = run_bench(sess_fp, feed, warmup, runs)
        mean_i8, min_i8, max_i8 = run_bench(sess_i8, feed, warmup, runs)

        speedup = mean_fp / mean_i8 if mean_i8 > 0 else float("inf")

        print(f"\n  ── Latence ─────────────────────────────────────────────────────────")
        print(f"  {'Modèle':<22}  {'moy':>8}  {'min':>8}  {'p100 (max)':>12}  {'taille':>10}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*10}")
        print(f"  {'FP32  (original)':<22}  {mean_fp:>7.1f}ms  {min_fp:>7.1f}ms  {max_fp:>11.1f}ms  {src_mb:>8.1f} MB")
        print(f"  {'INT8  (quantifié)':<22}  {mean_i8:>7.1f}ms  {min_i8:>7.1f}ms  {max_i8:>11.1f}ms  {dst_mb:>8.1f} MB")
        print(f"\n  → Accélération : {speedup:.2f}×  "
              f"(INT8 est {(1 - mean_i8 / mean_fp) * 100:.0f}% plus rapide en moyenne)")
    except Exception as e:
        print(f"  ⚠️   Benchmark latence échoué : {e}")
        print(f"       (les fichiers quantifiés sont valides — seule la mesure de perf a échoué)")

    # ── Qualité des prédictions ───────────────────────────────────────────────
    print(f"\n  ── Fidélité FP32 → INT8 ────────────────────────────────────────────")
    try:
        stats = compare_outputs(sess_fp, sess_i8, feed)
        for i, s in enumerate(stats):
            top1_str = f"  top-1 match={s['top1_match_pct']:.1f}%" if s["top1_match_pct"] is not None else ""
            print(f"  sortie[{i}]  MAE={s['mae']:.5f}  p99_err={s['p99']:.5f}{top1_str}")
            if s["top1_match_pct"] is not None:
                quality = "🟢 excellent" if s["top1_match_pct"] > 99 else \
                          "🟡 acceptable" if s["top1_match_pct"] > 95 else \
                          "🔴 dégradé"
                print(f"           → {quality}")
    except Exception as e:
        print(f"  ⚠️   Comparaison de sorties échouée : {e}")


# ============================================================================
# Lecture application.yml
# ============================================================================

def read_paths_from_yml(yml_path: str) -> tuple[Optional[str], Optional[str]]:
    if not _HAS_YAML:
        print("⚠️   PyYAML non installé — pip install pyyaml")
        return None, None
    with open(yml_path) as f:
        cfg = _yaml.safe_load(f)
    onnx = cfg.get("onnx", {})
    bilou = onnx.get("ner", {}).get("label", {}).get("modelPath")
    span  = onnx.get("ner", {}).get("ud",    {}).get("modelPath")
    return bilou, span


# ============================================================================
# main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    p.add_argument("--app-yml",      default=None,
                   help="Chemin vers application.yml (résolution auto des modèles)")
    p.add_argument("--bilou-model",  default=None, metavar="PATH",
                   help="Chemin explicite vers le modèle BILOU .onnx")
    p.add_argument("--span-model",   default=None, metavar="PATH",
                   help="Chemin explicite vers le modèle SpanNER .onnx")
    p.add_argument("--warmup",       type=int, default=5,  metavar="N",
                   help="Itérations de chauffe avant mesure (défaut: 5)")
    p.add_argument("--runs",         type=int, default=50, metavar="N",
                   help="Itérations de mesure (défaut: 50)")
    p.add_argument("--op-types",     nargs="+", default=["MatMul", "Gather"],
                   metavar="OP",
                   help="Op types ONNX à quantifier (défaut: MatMul Gather)")
    p.add_argument("--per-channel",  action="store_true",
                   help="Quantification per-channel (meilleure qualité, légèrement plus lent)")
    p.add_argument("--reduce-range", action="store_true",
                   help="Réduire la plage INT8 — pour CPU sans AVX-512 (vieux Intel)")
    p.add_argument("--no-bench",     action="store_true",
                   help="Quantifier sans lancer le benchmark de latence")
    p.add_argument("--only",         choices=["bilou", "span"], default=None,
                   help="Traiter uniquement un des deux modèles")
    args = p.parse_args()

    # ── Résolution des chemins ────────────────────────────────────────────────
    bilou_path: Optional[str] = args.bilou_model
    span_path:  Optional[str] = args.span_model

    if args.app_yml:
        b, s = read_paths_from_yml(args.app_yml)
        bilou_path = bilou_path or b
        span_path  = span_path  or s

    # Fallback : chemins relatifs habituels
    here = Path(__file__).parent
    bilou_path = bilou_path or str(here / "training_output" / "model_v2.onnx")
    span_path  = span_path  or str(here / "best_model_v3.onnx")

    # ── Résumé de la configuration ────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Quantification INT8 dynamique + Benchmark — modèles NER ONNX      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  BILOU  (NER label) : {bilou_path}")
    print(f"  SpanNER (UD)       : {span_path}")
    print(f"  Op types           : {args.op_types}")
    print(f"  Per-channel        : {args.per_channel}")
    print(f"  Reduce range       : {args.reduce_range}")
    if not args.no_bench:
        print(f"  Benchmark          : {args.warmup} warmup + {args.runs} runs")

    common = dict(
        op_types    =args.op_types,
        per_channel =args.per_channel,
        reduce_range=args.reduce_range,
        warmup      =args.warmup,
        runs        =args.runs,
        no_bench    =args.no_bench,
    )

    if args.only != "span":
        process_model(
            name     ="BILOU XLM-RoBERTa (ner.label)",
            src      =bilou_path,
            input_fn =bilou_inputs,
            **common,
        )

    if args.only != "bilou":
        process_model(
            name     ="SpanNER DeBERTa-v3 (ner.ud)",
            src      =span_path,
            input_fn =span_inputs,
            **common,
        )

    print("\n" + "═" * 72)
    print("  ✅  Terminé.")
    print("  Les modèles INT8 sont enregistrés à côté des originaux avec _int8.onnx")
    print()
    print("  Pour mettre à jour application.yml :")
    print(f"    onnx.ner.label.modelPath: {output_path(bilou_path)}")
    print(f"    onnx.ner.ud.modelPath:    {output_path(span_path)}")
    print("═" * 72)


if __name__ == "__main__":
    main()

