#!/usr/bin/env python3
"""
Export du checkpoint SpanMultiTaskModel vers ONNX via torch.onnx
(sans optimum, qui ne supporte pas les architectures custom).
Usage:
  python export_with_optimum.py \
      --checkpoint /path/to/checkpoint_best_multitask.pt \
      --output     /path/to/model.onnx \
      --model-name microsoft/deberta-v3-base
"""
import argparse, logging, os, warnings
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--opset",      type=int, default=17)
    args = parser.parse_args()

    from multitask_model import SpanMultiTaskModel
    from export_onnx_multitask import OnnxSpanMultiTaskWrapper

    print(f"📦 Chargement checkpoint : {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    inner = SpanMultiTaskModel(model_name=args.model_name).float()
    inner.load_state_dict(state)
    inner.eval()

    model = OnnxSpanMultiTaskWrapper(inner).eval()

    B, L, N = 1, 128, 64
    dummy_ids  = torch.zeros(B, L, dtype=torch.long)
    dummy_mask = torch.ones(B, L,  dtype=torch.long)
    dummy_ss   = (torch.arange(N) % (L - 2) + 1).long()
    dummy_se   = (dummy_ss + 1).clamp(max=L - 2)
    dummy_bid  = torch.zeros(N, dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"⚙️  Export ONNX opset={args.opset} → {args.output}")

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid),
        args.output,
        dynamo=False,
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask", "span_starts", "span_ends", "span_batch_ids"],
        output_names=["boundary_logits", "coarse_logits", "fine_logits"],
        dynamic_axes={
            "input_ids":       {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "seq_len"},
            "span_starts":     {0: "num_spans"},
            "span_ends":       {0: "num_spans"},
            "span_batch_ids":  {0: "num_spans"},
            "boundary_logits": {0: "num_spans"},
            "coarse_logits":   {0: "num_spans"},
            "fine_logits":     {0: "num_spans"},
        },
        do_constant_folding=False,
    )

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"✅ Export terminé — {size_mb:.1f} Mo")

    # Vérification rapide avec onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {
            "input_ids":       dummy_ids.numpy(),
            "attention_mask":  dummy_mask.numpy(),
            "span_starts":     dummy_ss.numpy(),
            "span_ends":       dummy_se.numpy(),
            "span_batch_ids":  dummy_bid.numpy(),
        })
        print(f"✅ Vérification ORT OK — boundary:{out[0].shape} coarse:{out[1].shape} fine:{out[2].shape}")
    except Exception as e:
        print(f"⚠️  Vérification ORT : {e}")

if __name__ == "__main__":
    main()

