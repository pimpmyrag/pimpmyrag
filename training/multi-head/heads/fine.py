"""Tête FINE — labels fins NER positive-only, masqués par coarse (soft-mask).

Couche propre : `model.fine_head` (Linear span_hidden_dim -> NUM_FINE).
Le masquage coarse->fine (fine_logits_masked, utilisé pour métriques/inférence)
reste calculé dans le modèle (nécessite coarse_fine_mask + coarse_logits d'une
autre tête = cross-head cascade), mais la loss et les métriques FINE sont ici.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import FINE_LABELS, FINE_CONCRETE_IDS, FINE_ABSTRACT_IDS
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


def _build_fine_diagnostics(y_true, y_pred, split_name: Optional[str] = None) -> dict:
    """Diagnostics fins exportables (confusions + labels difficiles)."""
    if not y_true:
        return {
            "fine_support_positive": 0,
            "fine_top_confusions": [],
            "fine_hard_labels": [],
        }

    label_names = list(FINE_LABELS) + ["INVALID_COARSE"]
    label_ids = list(range(len(FINE_LABELS))) + [-1]
    label_to_name = {i: n for i, n in zip(label_ids, label_names)}

    conf = {(t, p): 0 for t in label_ids for p in label_ids}
    for yt, yp in zip(y_true, y_pred):
        if yt not in label_to_name:
            continue
        pred_key = yp if yp in label_to_name else -1
        conf[(yt, pred_key)] += 1

    top_confusions = []
    for (yt, yp), count in conf.items():
        if count <= 0 or yt == yp:
            continue
        row_total = sum(conf[(yt, p)] for p in label_ids)
        top_confusions.append({
            "true_id": yt, "true_label": label_to_name[yt],
            "pred_id": yp, "pred_label": label_to_name[yp],
            "count": count,
            "row_pct": round(count / max(1, row_total), 4),
            "support": row_total,
        })
    top_confusions.sort(key=lambda x: (-x["count"], -x["row_pct"], x["true_label"], x["pred_label"]))

    hard_labels = []
    for label_id, label_name in enumerate(FINE_LABELS):
        row_total = sum(conf[(label_id, p)] for p in label_ids)
        if row_total <= 0:
            continue
        best_offdiag = max((conf[(label_id, p)], label_to_name[p]) for p in label_ids if p != label_id)
        tp = conf[(label_id, label_id)]
        recall = tp / row_total
        hard_labels.append({
            "label": label_name, "support": row_total, "recall": round(recall, 4),
            "top_confused_with": best_offdiag[1], "top_confused_count": best_offdiag[0],
        })
    hard_labels.sort(key=lambda x: (x["recall"], -x["support"], x["label"]))

    csv_path = json_path = None
    if split_name:
        sparse_rows = [
            {"true_label": label_to_name[yt], "pred_label": label_to_name[yp], "count": count}
            for (yt, yp), count in conf.items() if count > 0
        ]
        csv_path = f"fine_confusion_{split_name}.csv"
        json_path = f"fine_diagnostics_{split_name}.json"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["true_label", "pred_label", "count"])
            writer.writeheader()
            writer.writerows(sparse_rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"split": split_name, "top_confusions": top_confusions[:30], "hard_labels": hard_labels[:20]},
                f, ensure_ascii=False, indent=2,
            )

    return {
        "fine_support_positive": len(y_true),
        "fine_top_confusions": top_confusions[:20],
        "fine_hard_labels": hard_labels[:12],
        "fine_confusion_csv": csv_path,
        "fine_diagnostics_json": json_path,
    }


class FineHead(Head):
    task_key = "fine"
    jsonl_keys = {
        "fine_macro_f1": "fine_f1",
        "fine_concrete_f1": "fine_concrete_f1",
        "fine_abstract_f1": "fine_abstract_f1",
    }

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "FineHead":
        return cls(model.fine_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"fine_logits": self.linear(features["span_h_fine"])}

    def compute_loss(
        self,
        outputs,
        labels,
        sample_weights,
        class_weights: Optional[torch.Tensor] = None,
        focal_fine_gamma: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["fine_logits"].device
        f_logits = outputs["fine_logits"]
        boundary_labels = labels["boundary_labels"].to(device=device, dtype=torch.long)
        fine_labels = labels["fine_labels"].to(device=device, dtype=torch.long)

        pos_mask = (boundary_labels == 1) & (fine_labels >= 0) & (fine_labels < f_logits.size(-1))
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        loss_per = F.cross_entropy(f_logits[pos_mask], fine_labels[pos_mask],
                                    weight=class_weights, reduction="none")
        if focal_fine_gamma > 0.0:
            p_t = F.softmax(f_logits[pos_mask].detach(), dim=-1).gather(
                1, fine_labels[pos_mask].unsqueeze(1)).squeeze(1)
            loss_per = loss_per * (1.0 - p_t) ** focal_fine_gamma
        return (loss_per * sample_weights[pos_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        """Positive-only, prédiction via fine_logits_masked (déjà soft-masqué par coarse)."""
        f_pred = outputs["fine_logits_masked"].argmax(dim=-1).detach().cpu().tolist()
        f_true = gather_by_span_indices(labels["fine_labels"], span_indices)
        b_true = (context or {}).get("boundary_true")
        if b_true is None:
            return f_true, f_pred
        out_true, out_pred = [], []
        for bt, ft, fp in zip(b_true, f_true, f_pred):
            if bt == 1:
                out_true.append(ft)
                out_pred.append(fp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None, include_diagnostics: bool = True) -> dict:
        present = set(y_true)
        metrics = {
            "fine_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(len(FINE_LABELS)) if l in present]
            ),
            "fine_concrete_f1": safe_macro_f1(
                [l for l in y_true if l in FINE_CONCRETE_IDS],
                [p for l, p in zip(y_true, y_pred) if l in FINE_CONCRETE_IDS],
                labels=[l for l in FINE_CONCRETE_IDS if l in present],
            ) if any(l in FINE_CONCRETE_IDS for l in y_true) else 0.0,
            "fine_abstract_f1": safe_macro_f1(
                [l for l in y_true if l in FINE_ABSTRACT_IDS],
                [p for l, p in zip(y_true, y_pred) if l in FINE_ABSTRACT_IDS],
                labels=[l for l in FINE_ABSTRACT_IDS if l in present],
            ) if any(l in FINE_ABSTRACT_IDS for l in y_true) else 0.0,
            "fine_report": safe_report(
                y_true, y_pred, labels=list(range(len(FINE_LABELS))), target_names=FINE_LABELS
            ) if y_true else "N/A",
        }
        # include_diagnostics=False pendant le training : évite le coût CPU (confusions,
        # export CSV/JSON) — seulement calculé en eval (val/test), comme avant le refactor.
        if include_diagnostics:
            metrics.update(_build_fine_diagnostics(y_true, y_pred, split_name=split_name))
        return metrics

