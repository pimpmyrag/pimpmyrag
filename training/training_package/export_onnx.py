import os
import sys
import argparse
import torch
import torch.nn as nn

# allow running this script from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import SpanClassifier


def make_wrapper(model: SpanClassifier):
    """Create an nn.Module wrapper that exposes a forward suitable for ONNX export:
    Inputs:
      - input_ids:      LongTensor (B, L)
      - attention_mask: LongTensor (B, L)
      - span_starts:    LongTensor (S,)
      - span_ends:      LongTensor (S,)
      - span_batch_idx: LongTensor (S,)
      - coarse_ids:     LongTensor (S,)  ← famille NER coarse par span (0=PER … 5=OBJECT)
    Output:
      - logits: FloatTensor (S, num_labels)  — masqués par coarse (hard -1e9)
    """
    # ── Lookup table COARSE_TO_FINE : (6, 22) float ──────────────────────────
    # Construit comme une constante enregistrée dans le module → exportée dans le graphe ONNX.
    # Permet un masquage par simple indexation tensor[coarse_ids] sans boucle Python.
    _NEG = -1e4   # suffisamment grand après softmax, stable en FP16/INT8 (évite NaN avec -1e9)
    coarse_mask_data = torch.full((6, 22), _NEG)
    coarse_mask_data[0, [0, 1, 2, 3]] = 0.0              # PER   → person_name/role, norp, group_role
    coarse_mask_data[1, [5, 6, 7, 12]] = 0.0             # LOC   → gpe, fac_name, loc_generic, infra
    coarse_mask_data[2, [4]] = 0.0                        # ORG   → org_name
    coarse_mask_data[3, [18, 19, 20]] = 0.0              # TIME  → time_date, time_clock, time_duration
    coarse_mask_data[4, [16, 17]] = 0.0                  # EVENT → event_nominal, event_named
    coarse_mask_data[5, [8, 9, 10, 11, 13, 14, 15, 21]] = 0.0  # OBJECT

    class ONNXSpanWrapper(nn.Module):
        def __init__(self, core: SpanClassifier):
            super().__init__()
            self.core = core
            self.encoder      = core.encoder
            self.coarse_embed = core.coarse_embed
            self.classifier   = core.classifier
            # Constante exportée dans le graphe ONNX (pas une boucle Python → traçable)
            self.register_buffer("coarse_mask", coarse_mask_data)

        def forward(self, input_ids, attention_mask, span_starts, span_ends, span_batch_idx, coarse_ids):
            outputs     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, L, D)

            if last_hidden.dtype != self.classifier.weight.dtype:
                last_hidden = last_hidden.to(self.classifier.weight.dtype)

            B, L, D = last_hidden.shape

            # Prefix-sum pour calculer les mean-pool de span sans boucle (ONNX-compatible)
            zeros  = torch.zeros((B, 1, D), dtype=last_hidden.dtype, device=last_hidden.device)
            cumsum = torch.cumsum(last_hidden, dim=1)
            pref   = torch.cat([zeros, cumsum], dim=1)  # (B, L+1, D)

            span_starts    = span_starts.long()
            span_ends      = span_ends.long()
            span_batch_idx = span_batch_idx.long()
            coarse_ids     = coarse_ids.long()

            end_vecs   = pref[span_batch_idx, span_ends,   :]  # (S, D)
            start_vecs = pref[span_batch_idx, span_starts, :]  # (S, D)
            span_sums  = end_vecs - start_vecs
            lengths    = (span_ends - span_starts).unsqueeze(1).clamp(min=1).to(dtype=span_sums.dtype)
            span_means = span_sums / lengths  # (S, D)

            # Conditioning coarse : on concatène l'embedding de la famille NER
            coarse_vecs = self.coarse_embed(coarse_ids).to(dtype=span_means.dtype)  # (S, coarse_embed_dim)
            combined    = torch.cat([span_means, coarse_vecs], dim=-1)              # (S, D + coarse_embed_dim)

            logits = self.classifier(combined)  # (S, num_labels)

            # ── Masquage structurel coarse → fine (hard -1e4) ────────────────
            # self.coarse_mask : (6, 22) — lookup par coarse_id
            # mask[i] = 0.0 si label autorisé pour cette famille, -1e4 sinon
            mask = self.coarse_mask[coarse_ids].to(dtype=logits.dtype)  # (S, 22)
            return logits + mask

    return ONNXSpanWrapper(model)


def export(point_path: str, output_path: str, model_name: str = "microsoft/deberta-v3-base", device: str = "cpu", opset: int = 17):
    device_torch = torch.device(device)

    print(f"Loading model architecture ({model_name}) and point from: {point_path}")
    model = SpanClassifier(model_name, num_labels=22, num_coarse=6, coarse_embed_dim=128)
    # load state_dict
    state = torch.load(point_path, map_location='cpu')
    # training saved either state_dict or full ckpt
    if isinstance(state, dict) and 'model_state' in state:
        sd = state['model_state']
    else:
        sd = state
    model.load_state_dict(sd)
    model.to(device_torch)
    model.eval()

    wrapper = make_wrapper(model).to(device_torch)
    wrapper.eval()

    # build dummy inputs (small sizes) with proper dtypes
    B = 2
    L = 32
    S = 5

    dummy_input_ids = torch.randint(0, 1000, (B, L), dtype=torch.long, device=device_torch)
    dummy_attention = torch.ones((B, L), dtype=torch.long, device=device_torch)

    span_batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device_torch)
    span_starts    = torch.tensor([1, 5, 2, 7, 10], dtype=torch.long, device=device_torch)
    span_ends      = torch.tensor([3, 8, 4, 12, 15], dtype=torch.long, device=device_torch)
    # coarse_ids : famille NER coarse par span  (0=PER … 5=OBJECT)
    coarse_ids     = torch.tensor([0, 2, 1, 3, 5], dtype=torch.long, device=device_torch)

    input_names  = ["input_ids", "attention_mask", "span_starts", "span_ends", "span_batch_idx", "coarse_ids"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids":      {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "span_starts":    {0: "num_spans"},
        "span_ends":      {0: "num_spans"},
        "span_batch_idx": {0: "num_spans"},
        "coarse_ids":     {0: "num_spans"},
        "logits":         {0: "num_spans"},
    }

    print(f"Exporting ONNX to {output_path} (opset {opset}) -- this may take a moment...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention, span_starts, span_ends, span_batch_idx, coarse_ids),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    print("Export completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", default="best_model.pt", help="Path to point (state_dict or full ckpt)")
    parser.add_argument("--output", default="best_model.onnx", help="Output ONNX path")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base", help="Underlying HF model name used in training")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Device for export (cpu recommended)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export(args.point, args.output, model_name=args.model_name, device=args.device, opset=args.opset)
