from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import csv
import json
from collections import Counter

# W&B — import optionnel pour ne pas bloquer si non installé
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel
from loss_weighting import create_weighting, FixedWeighting, TASK_KEYS
from labels import (
    COARSE_LABELS, FINE_LABELS, COARSE_NONE_ID,
    SYN_LABELS, NUM_SYN,
    NUM_VOICE, NUM_CERTAINTY, CERTAINTY_LABELS, NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    ROLE_COARSE_LABELS, NUM_ROLE_COARSE, ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID,
    ROLE_OBLIQUE_LABELS, NUM_ROLE_OBLIQUE, ROLE_OBLIQUE_NONE_ID,
    ROLE_LABELS, NUM_ROLE, ROLE_NONE_ID,    # ancienne tête (12 labels)
    ROLE_COARSE2ID,
    PERSON_LABELS,
    FINE_CONCRETE_IDS, FINE_ABSTRACT_IDS,
    # VerbFam
    VERB_FAMILY_LABELS, NUM_VERB_FAMILY, VERB_FAMILY_NONE_ID,
    VERB_FAMILY_FINE_LABELS, NUM_VERB_FAMILY_FINE, VERB_FAMILY_FINE_NONE_ID,
    VERB_POLARITY_LABELS, NUM_VERB_POLARITY, VERB_POLARITY_NONE_ID,
    VERB_ASPECT_LABELS, NUM_VERB_ASPECT, VERB_ASPECT_NONE_ID,
    VERB_SOURCE_LABELS, NUM_VERB_SOURCE, VERB_SOURCE_NONE_ID,
    # compat
    SVO_LABELS, NUM_SVO,
)

# ──────────────────────────────────────────────────────────
#  Inline Hard Negative Mining — constantes
# ──────────────────────────────────────────────────────────
_LOW_PRECISION_COARSE = {"VALUE", "EVENT", "TIME", "ABSTRACT", "WORK"}
_LOW_F1_FINE = {
    "hint_measure", "hint_rate", "hint_infra", "hint_object_generic",
    "hint_inst_name", "hint_document",   # nouveaux labels v5 — potentiellement rares en début de training
}
_FP_LOW_PREC_EXTRA = 1.5
_FINE_ERR_EXTRA    = 1.4


# ──────────────────────────────────────────────────────────
#  EMA — Exponential Moving Average des poids du modèle
# ──────────────────────────────────────────────────────────
class ModelEMA:
    """
    Maintient une copie lissée des poids du modèle.
    Gain typique : +0.5 à +1.5% sur val score sans autre changement.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().float().detach()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float().detach()

    def apply(self, model: torch.nn.Module) -> dict:
        """Injecte les poids EMA dans le modèle. Retourne l'état original."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        ema_state = {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()}
        model.load_state_dict(ema_state)
        return original

    def restore(self, model: torch.nn.Module, original_state: dict):
        model.load_state_dict(original_state)

    def state_dict(self):
        return self.shadow


def build_fine_diagnostics(y_true, y_pred, split_name: str | None = None) -> dict:
    """Construit des diagnostics fins légers et exportables.

    - inclut un bucket `INVALID_COARSE` pour les prédictions fine = -1
    - exporte une matrice sparse CSV et un résumé JSON si `split_name` est fourni
    """
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
    support = Counter(y_true)
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
            "true_id": yt,
            "true_label": label_to_name[yt],
            "pred_id": yp,
            "pred_label": label_to_name[yp],
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
            "label": label_name,
            "support": row_total,
            "recall": round(recall, 4),
            "top_confused_with": best_offdiag[1],
            "top_confused_count": best_offdiag[0],
        })
    hard_labels.sort(key=lambda x: (x["recall"], -x["support"], x["label"]))

    if split_name:
        sparse_rows = [
            {
                "true_label": label_to_name[yt],
                "pred_label": label_to_name[yp],
                "count": count,
            }
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
                {
                    "split": split_name,
                    "top_confusions": top_confusions[:30],
                    "hard_labels": hard_labels[:20],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    else:
        csv_path = None
        json_path = None

    return {
        "fine_support_positive": len(y_true),
        "fine_top_confusions": top_confusions[:20],
        "fine_hard_labels": hard_labels[:12],
        "fine_confusion_csv": csv_path,
        "fine_diagnostics_json": json_path,
    }


# ──────────────────────────────────────────────────────────
#  Layer-wise LR decay
# ──────────────────────────────────────────────────────────
def get_layerwise_param_groups(model, base_lr: float, head_lr: float, decay: float = 0.9):
    """
    Crée des groupes de paramètres avec LR décroissant par couche.
    - Têtes (boundary/coarse/fine/MLP) : head_lr
    - Couche transformer i (depuis le haut) : base_lr * decay^(num_layers - 1 - i)
    - Embeddings : base_lr * decay^num_layers  (LR le plus faible)

    decay=1.0 → pas de decay (comportement original).
    """
    # Tous les paramètres hors-encodeur (heads, span_mlp, width_emb, etc.) → head_lr
    # On utilise model.parameters() - model.encoder.parameters() pour ne rien oublier
    # (les listes manuelles oublient facilement les nouvelles têtes comme verbfam/role)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    seen = {id(p) for p in head_params}
    param_groups = [{"params": head_params, "lr": head_lr, "name": "heads"}]

    if decay >= 1.0:
        enc_params = [p for p in model.encoder.parameters() if id(p) not in seen]
        param_groups.append({"params": enc_params, "lr": base_lr, "name": "encoder"})
        return param_groups

    encoder = model.encoder
    # DeBERTa-v3 : encoder.encoder.layer
    try:
        layers = encoder.encoder.layer
    except AttributeError:
        enc_params = [p for p in encoder.parameters() if id(p) not in seen]
        param_groups.append({"params": enc_params, "lr": base_lr, "name": "encoder"})
        return param_groups

    num_layers = len(layers)

    # Embeddings — LR le plus faible
    try:
        emb_params = [p for p in encoder.embeddings.parameters() if id(p) not in seen]
        if emb_params:
            emb_lr = base_lr * (decay ** num_layers)
            param_groups.append({"params": emb_params, "lr": emb_lr, "name": "embeddings"})
            seen.update(id(p) for p in emb_params)
    except AttributeError:
        pass

    # Couches transformer — decay croissant vers le bas
    for i, layer in enumerate(layers):
        layer_lr = base_lr * (decay ** (num_layers - 1 - i))
        layer_p = [p for p in layer.parameters() if id(p) not in seen]
        if layer_p:
            param_groups.append({"params": layer_p, "lr": layer_lr, "name": f"layer_{i}"})
            seen.update(id(p) for p in layer_p)

    # Reste (pooler, etc.)
    remaining = [p for p in encoder.parameters() if id(p) not in seen]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr, "name": "encoder_other"})

    return param_groups


def compute_class_weights_from_multitask_jsonl(path: str, power: float = 0.5):
    """
    Calcule des poids de classes à partir du dataset multitask enrichi.
    Retourne weights pour boundary, coarse, fine, certainty, role_oblique, role_coarse.
    """
    boundary_counts = Counter()
    coarse_counts = Counter()
    fine_counts = Counter()
    certainty_counts = Counter()
    oblique_counts = Counter()
    role_coarse_counts = Counter()
    verb_family_counts = Counter()
    verb_polarity_counts = Counter()
    verb_aspect_counts = Counter()
    verb_source_counts = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for c in row["candidates"]:
                boundary_counts[c["boundary_label"]] += 1
                coarse_counts[c["coarse_label_id"]] += 1
                fine_counts[c["fine_label_id"]] += 1
                cert_id = c.get("certainty_label_id", -1)
                if cert_id >= 0:
                    certainty_counts[cert_id] += 1
                obl_id = c.get("role_oblique_label_id", ROLE_OBLIQUE_NONE_ID)
                if obl_id < ROLE_OBLIQUE_NONE_ID:
                    oblique_counts[obl_id] += 1
                # role_coarse : compter seulement les vrais rôles (SUBJ/OBJ/OBLIQ/APPOS)
                # OTHER(4) et NONE_ID(5) exclus du gradient → pas dans les weights
                rc_id = c.get("role_coarse_label_id", ROLE_COARSE_NONE_ID)
                if rc_id < ROLE_COARSE_OTHER_ID:  # 0..3 seulement
                    role_coarse_counts[rc_id] += 1
                # verbfam : compter seulement les spans avec label valide (≠ NONE_ID)
                vf = c.get("verb_family_label_id", VERB_FAMILY_NONE_ID)
                if vf < VERB_FAMILY_NONE_ID:
                    verb_family_counts[vf] += 1
                vp = c.get("verb_polarity_label_id", VERB_POLARITY_NONE_ID)
                if vp < VERB_POLARITY_NONE_ID:
                    verb_polarity_counts[vp] += 1
                va = c.get("verb_aspect_label_id", VERB_ASPECT_NONE_ID)
                if va < VERB_ASPECT_NONE_ID:
                    verb_aspect_counts[va] += 1
                vs = c.get("verb_source_label_id", VERB_SOURCE_NONE_ID)
                if vs < VERB_SOURCE_NONE_ID:
                    verb_source_counts[vs] += 1

    def make_weights(counts, num_classes, power=0.5, max_weight=None):
        total = sum(counts.values())
        weights = torch.ones(num_classes, dtype=torch.float32)
        if total == 0:
            return weights
        for i in range(num_classes):
            n_i = counts.get(i, 0)
            if n_i > 0:
                inv_freq = total / (num_classes * n_i)
                w = float(inv_freq) ** float(power)
                # Plafond avant normalisation : évite que classes ultra-rares gonflent
                # la moyenne et écrasent les classes fréquentes (OBL_TIME/OBL_LOC)
                if max_weight is not None:
                    w = min(w, float(max_weight))
                weights[i] = w
            else:
                weights[i] = 1.0
        weights = weights / weights.mean()
        return weights

    # role_coarse : poids ÉGAUX (power=0) pour SUBJ/OBJ/OBLIQ/APPOS.
    # OBLIQ est la plus fréquente → avec power>0 elle reçoit le poids le plus bas → collapse SUBJ.
    # Equal weights → gradient neutre entre les 4 classes, le modèle apprend toutes simultanément.
    # OTHER(4) reçoit poids=1.0 (neutre, présent dans softmax mais exclu du gradient)
    rc_w = make_weights(role_coarse_counts, ROLE_COARSE_OTHER_ID, power=0.0)
    rc_w_full = torch.ones(NUM_ROLE_COARSE, dtype=torch.float32)
    rc_w_full[:ROLE_COARSE_OTHER_ID] = rc_w  # indices 0-3 = SUBJ/OBJ/OBLIQ/APPOS
    # index 4 (OTHER) = 1.0 neutre (exclu du gradient de toute façon)

    return (
        make_weights(boundary_counts, 2, power=power),
        make_weights(coarse_counts, len(COARSE_LABELS), power=0.0),
        make_weights(fine_counts, len(FINE_LABELS), power=power),
        make_weights(certainty_counts, NUM_CERTAINTY, power=power),
        # max_weight=3.0 : plafonne les classes ultra-rares (OBL_COMITATIVE ~0.7%, weight brut ~15x)
        # avant normalisation par la moyenne — sans ce plafond, OBL_TIME/OBL_LOC (16%/25%)
        # obtiennent weight_final < 0.1 (quasi éliminés) car la moyenne est gonflée par les rares.
        make_weights(oblique_counts, NUM_ROLE_OBLIQUE, power=power, max_weight=3.0),
        rc_w_full,
        # verbfam class weights — power plus modéré pour ne pas écraser State_Change (25.9%)
        # power=min(power, 0.5) au lieu de power*1.5 : Conflict 2x boost, State_Change ~0.57 (pas 0.31)
        make_weights(verb_family_counts, NUM_VERB_FAMILY, power=min(power, 0.5)),
        make_weights(verb_polarity_counts, NUM_VERB_POLARITY, power=power),
        make_weights(verb_aspect_counts, NUM_VERB_ASPECT, power=min(power * 1.2, 0.7)),
        make_weights(verb_source_counts, NUM_VERB_SOURCE, power=power),
        boundary_counts,
        coarse_counts,
        fine_counts,
        certainty_counts,
        oblique_counts,
        role_coarse_counts,
    )


def apply_class_weight_floor(weights: torch.Tensor, class_idx: int, min_value: float) -> torch.Tensor:
    """Applique un plancher minimal à une classe sans re-normaliser ensuite."""
    weights = weights.clone()
    weights[class_idx] = max(weights[class_idx].item(), float(min_value))
    return weights


def safe_macro_f1(y_true, y_pred, labels=None):
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def fine_pred_from_masked(fine_logits_masked, c_pred):
    """Prédit les labels fine à partir des logits déjà soft-masqués par coarse (forward).
    c_pred non utilisé car le masquage est déjà appliqué dans forward() — argmax suffit."""
    return fine_logits_masked.argmax(dim=-1).detach().cpu().tolist()


def run_epoch(
        loader,
        model,
        optimizer,
        device,
        train: bool,
        boundary_class_weights=None,
        coarse_class_weights=None,
        fine_class_weights=None,
        certainty_class_weights=None,
        role_coarse_class_weights=None,
        oblique_class_weights=None,
        verb_family_class_weights=None,
        verb_polarity_class_weights=None,
        verb_aspect_class_weights=None,
        verb_source_class_weights=None,
        lambda_boundary=2.5,
        lambda_coarse=0.5,
        lambda_fine=1.0,
        lambda_svo_boundary=0.5,
        lambda_svo=0.5,
        lambda_role_coarse=0.0,
        lambda_role_oblique=0.3,
        lambda_role=0.5,
        lambda_voice=0.15,
        lambda_certainty=0.4,
        lambda_morpho=0.3,
        lambda_verb_ptr=0.25,
        lambda_compat=0.0,
        lambda_verb_family=0.0,
        lambda_verb_family_fine=0.0,
        lambda_verb_polarity=0.0,
        lambda_verb_aspect=0.0,
        lambda_verb_source=0.0,
        accum_steps=1,
        log_every=50,
        focal_gamma=0.0,
        max_grad_norm=1.0,
        ema: "ModelEMA | None" = None,
        collect_hn: bool = False,
        scaler=None,           # torch.GradScaler pour AMP (None = FP32)
        eval_split: str | None = None,
        focal_fine_gamma: float = 0.0,   # Focal loss sur tête fine
        focal_coarse_gamma: float = 0.0, # Focal loss sur tête coarse (positifs seulement)
        focal_role_gamma: float = 0.0,   # Focal loss sur tête role
        ignore_coarse_none: bool = False, # Si True, exclut spans NONE de la loss coarse
        weighting=None,  # Dynamic loss weighting module
        gradnorm_every: int = 10,  # GradNorm update frequency (optimizer steps)
):
    """
    Version adaptée à l'architecture :
      - boundary : binaire
      - coarse   : 6 familles + NONE
      - fine     : 22 labels positifs uniquement
      - fine loss : positive-only
      - fine metrics : positive-only
      - fine prediction : masquage coarse -> fine

    Retourne:
      metrics = {
          "loss": float,
          "boundary_f1": float,
          "coarse_macro_f1": float,
          "fine_macro_f1": float,              # positive-only
          "boundary_report": str,
          "coarse_report": str,
          "fine_report": str,                  # positive-only
      }
    """

    import torch
    from sklearn.metrics import f1_score, classification_report

    def safe_macro_f1_local(y_true, y_pred, labels=None):
        if not y_true:
            return 0.0
        return f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=labels,
            zero_division=0
        )

    def masked_fine_predictions(fine_logits, coarse_preds, coarse_fine_mask):
        """
        Applique un masquage coarse -> fine.

        Args:
            fine_logits: [N, F]
            coarse_preds: [N]
            coarse_fine_mask: [C, F] bool

        Returns:
            pred_fine: list[int] de taille N
                - label fine prédit si coarse != NONE et masque valide
                - -1 si coarse = NONE ou aucun fine autorisé
        """
        if fine_logits.numel() == 0:
            return []

        device_local = fine_logits.device
        coarse_preds_t = torch.as_tensor(coarse_preds, dtype=torch.long, device=device_local)

        # [N, F]
        allowed = coarse_fine_mask[coarse_preds_t]

        # rows sans aucun label fine autorisé (ex: coarse=NONE)
        no_valid = ~allowed.any(dim=-1)

        masked_logits = fine_logits.clone()
        fill_val = torch.finfo(masked_logits.dtype).min
        masked_logits = masked_logits.masked_fill(~allowed, fill_val)

        pred = masked_logits.argmax(dim=-1)

        # pour les coarse=NONE / rows invalides, on force une prédiction invalide
        pred = pred.masked_fill(no_valid, -1)

        return pred.detach().cpu().tolist()

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    losses = []

    all_b_true, all_b_pred = [], []
    all_c_true, all_c_pred = [], []

    # fine positive-only
    all_f_true_pos, all_f_pred_pos = [], []

    # coarse positive-only (boundary=1 uniquement pour les métriques)
    all_c_true_pos, all_c_pred_pos = [], []

    # svo_boundary
    all_svob_true, all_svob_pred = [], []
    # role_coarse positive-only (spans avec role_coarse != ROLE_COARSE_NONE_ID)
    all_rc_true, all_rc_pred = [], []
    # role_coarse dérivée depuis role_head (même mask — diagnostic comparatif)
    all_rc_from_role_true, all_rc_from_role_pred = [], []
    # role_oblique positive-only (spans avec role_oblique < ROLE_OBLIQUE_NONE_ID)
    all_ro_true, all_ro_pred = [], []
    # role_oblique CASCADE — gate = role_coarse_from_role prédit OBLIQ (mode inférence réelle)
    all_ro_cascaded_true, all_ro_cascaded_pred = [], []
    # role fin (12 labels) : spans avec rôle annoté (< ROLE_NONE_ID)
    all_role_true, all_role_pred = [], []
    # voice positive-only (spans avec voice_label != VOICE_NONE_ID)
    all_voice_true, all_voice_pred = [], []
    # certainty positive-only (spans avec certainty_label != CERTAINTY_NONE_ID)
    all_certainty_true, all_certainty_pred = [], []
    # morpho positive-only (spans SVO actifs)
    all_gender_true, all_gender_pred = [], []
    all_number_true, all_number_pred = [], []
    all_person_true, all_person_pred = [], []
    # verb pointer : accuracy sur spans avec gov_verb_labels >= 0
    all_ptr_true, all_ptr_pred = [], []

    # VerbFam (verb_trigger uniquement)
    all_vfam_true,      all_vfam_pred      = [], []
    all_vfam_fine_true, all_vfam_fine_pred = [], []
    all_vpol_true,      all_vpol_pred      = [], []
    all_vasp_true,      all_vasp_pred      = [], []
    all_vsrc_true,      all_vsrc_pred      = [], []

    # inline HN mining : id → list[(err_type|None, pred_coarse, pred_fine)]
    hn_results_by_id: dict[str, list] = {} if collect_hn else None

    coarse_fine_mask = getattr(model, "coarse_fine_mask", None)
    if coarse_fine_mask is None:
        raise ValueError(
            "Le modèle n'expose pas `coarse_fine_mask`. "
            "Assure-toi d'utiliser la version modifiée de multitask_model.py."
        )

    coarse_fine_mask = coarse_fine_mask.to(device)

    amp_enabled = (device == "cuda") and (scaler is not None)

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        spans = batch["spans"]

        boundary_labels = batch["boundary_labels"].to(device)
        coarse_labels = batch["coarse_labels"].to(device)
        fine_labels = batch["fine_labels"].to(device)
        svo_boundary_labels = batch["svo_boundary_labels"].to(device)
        syn_labels    = batch["syn_labels"].to(device)
        role_coarse_labels = batch.get("role_coarse_labels")
        if role_coarse_labels is not None:
            role_coarse_labels = role_coarse_labels.to(device)
        else:
            role_coarse_labels = torch.full_like(syn_labels, ROLE_COARSE_NONE_ID)
        role_oblique_labels = batch.get("role_oblique_labels")
        if role_oblique_labels is not None:
            role_oblique_labels = role_oblique_labels.to(device)
        else:
            role_oblique_labels = torch.full_like(syn_labels, ROLE_OBLIQUE_NONE_ID)
        role_labels = batch.get("role_labels")
        if role_labels is not None:
            role_labels = role_labels.to(device)
        else:
            role_labels = torch.full_like(syn_labels, ROLE_NONE_ID)
        voice_labels  = batch["voice_labels"].to(device)
        certainty_labels = batch.get("certainty_labels")
        if certainty_labels is not None:
            certainty_labels = certainty_labels.to(device)
        else:
            from labels import CERTAINTY_NONE_ID
            certainty_labels = torch.full_like(syn_labels, CERTAINTY_NONE_ID)
        gender_labels = batch["gender_labels"].to(device)
        number_labels = batch["number_labels"].to(device)
        person_labels = batch["person_labels"].to(device)
        gov_verb_labels = batch["gov_verb_labels"].to(device)
        sample_weights = batch["sample_weights"].to(device)
        # VerbFam labels (optionnels pour compat datasets anciens)
        verb_family_labels      = batch.get("verb_family_labels")
        verb_family_fine_labels = batch.get("verb_family_fine_labels")
        verb_polarity_labels    = batch.get("verb_polarity_labels")
        verb_aspect_labels      = batch.get("verb_aspect_labels")
        verb_source_labels      = batch.get("verb_source_labels")
        if verb_family_labels is not None:
            verb_family_labels      = verb_family_labels.to(device)
            verb_family_fine_labels = verb_family_fine_labels.to(device)
            verb_polarity_labels    = verb_polarity_labels.to(device)
            verb_aspect_labels      = verb_aspect_labels.to(device)
            verb_source_labels      = verb_source_labels.to(device)

        # Sanity check avant forward
        num_spans = sum(len(x) for x in spans)
        if not (
                num_spans
                == boundary_labels.size(0)
                == coarse_labels.size(0)
                == fine_labels.size(0)
                == svo_boundary_labels.size(0)
                == syn_labels.size(0)
                == voice_labels.size(0)
                == role_coarse_labels.size(0)
                == certainty_labels.size(0)
                == gender_labels.size(0)
                == number_labels.size(0)
                == person_labels.size(0)
                == gov_verb_labels.size(0)
                == sample_weights.size(0)
        ):
            raise ValueError(
                "Mismatch batch avant forward: "
                f"num_spans={num_spans}, "
                f"boundary={boundary_labels.size(0)}, "
                f"coarse={coarse_labels.size(0)}, "
                f"fine={fine_labels.size(0)}, "
                f"sample_weights={sample_weights.size(0)}, "
                f"ids={batch.get('ids')}"
            )

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                outputs = model({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "spans": spans,
                })

                # Si le modele renvoie span_indices, on aligne tout sur les spans gardes
                span_indices = outputs.get("span_indices", None)
                if span_indices is not None:
                    si = span_indices.to(device=device, dtype=torch.long)
                    boundary_labels_loss     = boundary_labels[si]
                    coarse_labels_loss       = coarse_labels[si]
                    fine_labels_loss         = fine_labels[si]
                    svo_boundary_labels_loss = svo_boundary_labels[si]
                    syn_labels_loss          = syn_labels[si]
                    role_coarse_labels_loss  = role_coarse_labels[si]
                    role_oblique_labels_loss = role_oblique_labels[si]
                    role_labels_loss         = role_labels[si]
                    voice_labels_loss        = voice_labels[si]
                    certainty_labels_loss    = certainty_labels[si]
                    gender_labels_loss       = gender_labels[si]
                    number_labels_loss       = number_labels[si]
                    person_labels_loss       = person_labels[si]
                    gov_verb_labels_loss     = gov_verb_labels[si]
                    sample_weights_loss      = sample_weights[si]
                    verb_family_labels_loss      = verb_family_labels[si]      if verb_family_labels is not None else None
                    verb_family_fine_labels_loss = verb_family_fine_labels[si] if verb_family_fine_labels is not None else None
                    verb_polarity_labels_loss    = verb_polarity_labels[si]    if verb_polarity_labels is not None else None
                    verb_aspect_labels_loss      = verb_aspect_labels[si]      if verb_aspect_labels is not None else None
                    verb_source_labels_loss      = verb_source_labels[si]      if verb_source_labels is not None else None
                else:
                    boundary_labels_loss     = boundary_labels
                    coarse_labels_loss       = coarse_labels
                    fine_labels_loss         = fine_labels
                    svo_boundary_labels_loss = svo_boundary_labels
                    syn_labels_loss          = syn_labels
                    role_coarse_labels_loss  = role_coarse_labels
                    role_oblique_labels_loss = role_oblique_labels
                    role_labels_loss         = role_labels
                    voice_labels_loss        = voice_labels
                    certainty_labels_loss    = certainty_labels
                    gender_labels_loss       = gender_labels
                    number_labels_loss       = number_labels
                    person_labels_loss       = person_labels
                    gov_verb_labels_loss     = gov_verb_labels
                    sample_weights_loss      = sample_weights
                    verb_family_labels_loss      = verb_family_labels
                    verb_family_fine_labels_loss = verb_family_fine_labels
                    verb_polarity_labels_loss    = verb_polarity_labels
                    verb_aspect_labels_loss      = verb_aspect_labels
                    verb_source_labels_loss      = verb_source_labels

                # Sanity check apres forward / avant loss
                num_logits = outputs["fine_logits"].size(0)
                if not (
                        num_logits
                        == boundary_labels_loss.size(0)
                        == coarse_labels_loss.size(0)
                        == fine_labels_loss.size(0)
                        == sample_weights_loss.size(0)
                ):
                    raise ValueError(
                        "Mismatch logits/labels apres forward: "
                        f"logits={num_logits}, "
                        f"boundary={boundary_labels_loss.size(0)}, "
                        f"coarse={coarse_labels_loss.size(0)}, "
                        f"fine={fine_labels_loss.size(0)}, "
                        f"sample_weights={sample_weights_loss.size(0)}, "
                        f"ids={batch.get('ids')}"
                    )

                loss_dict = model.compute_loss(
                    outputs=outputs,
                    boundary_labels=boundary_labels_loss,
                    coarse_labels=coarse_labels_loss,
                    fine_labels=fine_labels_loss,
                    svo_boundary_labels=svo_boundary_labels_loss,
                    syn_labels=syn_labels_loss,
                    role_coarse_labels=role_coarse_labels_loss,
                    role_oblique_labels=role_oblique_labels_loss,
                    role_labels=role_labels_loss,
                    voice_labels=voice_labels_loss,
                    certainty_labels=certainty_labels_loss,
                    gender_labels=gender_labels_loss,
                    number_labels=number_labels_loss,
                    person_labels=person_labels_loss,
                    gov_verb_labels=gov_verb_labels_loss,
                    sample_weights=sample_weights_loss,
                    # verbfam (optionnel)
                    verb_family_labels=verb_family_labels_loss,
                    verb_family_fine_labels=verb_family_fine_labels_loss,
                    verb_polarity_labels=verb_polarity_labels_loss,
                    verb_aspect_labels=verb_aspect_labels_loss,
                    verb_source_labels=verb_source_labels_loss,
                    boundary_class_weights=boundary_class_weights,
                    coarse_class_weights=coarse_class_weights,
                    fine_class_weights=fine_class_weights,
                    certainty_class_weights=certainty_class_weights,
                    oblique_class_weights=oblique_class_weights,
                    role_coarse_class_weights=role_coarse_class_weights,
                    verb_family_class_weights=verb_family_class_weights,
                    verb_polarity_class_weights=verb_polarity_class_weights,
                    verb_aspect_class_weights=verb_aspect_class_weights,
                    verb_source_class_weights=verb_source_class_weights,
                    lambda_boundary=lambda_boundary,
                    lambda_coarse=lambda_coarse,
                    lambda_fine=lambda_fine,
                    lambda_svo_boundary=lambda_svo_boundary,
                    lambda_svo=lambda_svo,
                    lambda_role_coarse=lambda_role_coarse,
                    lambda_role_oblique=lambda_role_oblique,
                    lambda_role=lambda_role,
                    lambda_voice=lambda_voice,
                    lambda_certainty=lambda_certainty,
                    lambda_morpho=lambda_morpho,
                    lambda_verb_ptr=lambda_verb_ptr,
                    lambda_compat=lambda_compat,
                    lambda_verb_family=lambda_verb_family,
                    lambda_verb_family_fine=lambda_verb_family_fine,
                    lambda_verb_polarity=lambda_verb_polarity,
                    lambda_verb_aspect=lambda_verb_aspect,
                    lambda_verb_source=lambda_verb_source,
                    focal_gamma=focal_gamma,
                    focal_coarse_gamma=focal_coarse_gamma,
                    focal_fine_gamma=focal_fine_gamma,
                    focal_role_gamma=focal_role_gamma,
                    ignore_coarse_none=ignore_coarse_none,
                    weighting=weighting,
                )

                loss = loss_dict["loss"] / accum_steps

            if train:
                # ── GradNorm step BEFORE main backward (needs the graph alive) ──
                _gn_step_id = getattr(run_epoch, '_gn_step_counter', 0) + 1
                run_epoch._gn_step_counter = _gn_step_id
                if (weighting is not None
                        and hasattr(weighting, 'gradnorm_loss')
                        and _gn_step_id % gradnorm_every == 0
                        and step % accum_steps == 0):
                    from loss_weighting import GradNormWeighting
                    if isinstance(weighting, GradNormWeighting):
                        ramp_lambdas = {
                            "boundary": lambda_boundary, "coarse": lambda_coarse,
                            "fine": lambda_fine, "svo_boundary": lambda_svo_boundary,
                            "svo": lambda_svo,
                            "voice": lambda_voice, "certainty": lambda_certainty,
                            "morpho": lambda_morpho, "verb_ptr": lambda_verb_ptr,
                            "compat": lambda_compat,
                        }
                        raw = loss_dict.get("raw_losses")
                        if raw:
                            gn_loss = weighting.gradnorm_loss(raw, model.span_mlp, ramp_lambdas)
                            if gn_loss is not None:
                                gn_loss.backward(retain_graph=True)
                                # Manual step on weighting params only
                                for pg in optimizer.param_groups:
                                    if pg.get("name") == "loss_weighting":
                                        for p in pg["params"]:
                                            if p.grad is not None:
                                                p.data -= pg["lr"] * p.grad
                                                p.grad = None
                                weighting.renormalize(ramp_lambdas)

                # ── Main backward ──
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        if train and (step % accum_steps == 0):
            if max_grad_norm > 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        losses.append(loss_dict["loss"].item())

        if (step % log_every == 0) or (step == 1):
            mode = "TRAIN" if train else "EVAL"
            try:
                total_steps = len(loader)
            except Exception:
                total_steps = "?"
            avg_loss = sum(losses) / max(1, len(losses))
            compat_val = loss_dict.get("loss_compat", None)
            compat_str = f" compat={compat_val.item():.4f}" if compat_val is not None else ""
            print(
                f"[{mode}] step={step}/{total_steps} "
                f"loss={loss_dict['loss'].item():.4f} "
                f"avg_loss={avg_loss:.4f}"
                f"{compat_str}"
            )

        # Predictions
        b_pred = outputs["boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        c_pred = outputs["coarse_logits"].argmax(dim=-1).detach().cpu().tolist()
        # fine_logits_masked : déjà soft-masqué par coarse dans forward() → même chemin qu'à l'inférence
        f_pred = fine_pred_from_masked(outputs["fine_logits_masked"], c_pred)
        svob_pred    = outputs["svo_boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        role_coarse_pred_raw = outputs["role_coarse_logits"].argmax(dim=-1).detach().cpu().tolist()
        role_coarse_from_role_pred_raw = outputs["role_coarse_from_role_logits"].argmax(dim=-1).detach().cpu().tolist()
        role_oblique_pred_raw = outputs["role_oblique_logits"].argmax(dim=-1).detach().cpu().tolist()
        role_pred_raw = outputs["role_logits"].argmax(dim=-1).detach().cpu().tolist()
        voice_pred_raw = outputs["voice_logits"].argmax(dim=-1).detach().cpu().tolist()
        certainty_pred_raw = outputs["certainty_logits"].argmax(dim=-1).detach().cpu().tolist()
        gender_pred_raw = outputs["gender_logits"].argmax(dim=-1).detach().cpu().tolist()
        number_pred_raw = outputs["number_logits"].argmax(dim=-1).detach().cpu().tolist()
        person_pred_raw = outputs["person_logits"].argmax(dim=-1).detach().cpu().tolist()
        # verb pointer : argmax sur la dim seq pour chaque span
        vptr_logits_cpu = outputs["verb_ptr_logits"].detach().cpu()  # [N, seq]
        ptr_pred_raw = vptr_logits_cpu.argmax(dim=-1).tolist()       # [N]
        # VerbFam heads
        vfam_pred_raw      = outputs["verb_family_logits"].argmax(dim=-1).detach().cpu().tolist()
        vfam_fine_pred_raw = outputs["verb_family_fine_logits"].argmax(dim=-1).detach().cpu().tolist()
        vpol_pred_raw      = outputs["verb_polarity_logits"].argmax(dim=-1).detach().cpu().tolist()
        vasp_pred_raw      = outputs["verb_aspect_logits"].argmax(dim=-1).detach().cpu().tolist()
        vsrc_pred_raw      = outputs["verb_source_logits"].argmax(dim=-1).detach().cpu().tolist()

        # Vérité terrain alignée sur les spans scorés
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            b_true    = boundary_labels.detach().cpu()[si_cpu].tolist()
            c_true    = coarse_labels.detach().cpu()[si_cpu].tolist()
            f_true    = fine_labels.detach().cpu()[si_cpu].tolist()
            svob_true = svo_boundary_labels.detach().cpu()[si_cpu].tolist()
            role_coarse_true = role_coarse_labels.detach().cpu()[si_cpu].tolist()
            role_oblique_true = role_oblique_labels.detach().cpu()[si_cpu].tolist()
            role_true    = role_labels.detach().cpu()[si_cpu].tolist()
            voice_true   = voice_labels.detach().cpu()[si_cpu].tolist()
            certainty_true = certainty_labels.detach().cpu()[si_cpu].tolist()
            gender_true  = gender_labels.detach().cpu()[si_cpu].tolist()
            number_true  = number_labels.detach().cpu()[si_cpu].tolist()
            person_true  = person_labels.detach().cpu()[si_cpu].tolist()
            gov_verb_true = gov_verb_labels.detach().cpu()[si_cpu].tolist()
            vfam_true      = verb_family_labels.detach().cpu()[si_cpu].tolist()      if verb_family_labels is not None else [VERB_FAMILY_NONE_ID] * len(si_cpu)
            vfam_fine_true = verb_family_fine_labels.detach().cpu()[si_cpu].tolist() if verb_family_fine_labels is not None else [VERB_FAMILY_FINE_NONE_ID] * len(si_cpu)
            vpol_true      = verb_polarity_labels.detach().cpu()[si_cpu].tolist()    if verb_polarity_labels is not None else [VERB_POLARITY_NONE_ID] * len(si_cpu)
            vasp_true      = verb_aspect_labels.detach().cpu()[si_cpu].tolist()      if verb_aspect_labels is not None else [VERB_ASPECT_NONE_ID] * len(si_cpu)
            vsrc_true      = verb_source_labels.detach().cpu()[si_cpu].tolist()      if verb_source_labels is not None else [VERB_SOURCE_NONE_ID] * len(si_cpu)
        else:
            b_true    = boundary_labels.detach().cpu().tolist()
            c_true    = coarse_labels.detach().cpu().tolist()
            f_true    = fine_labels.detach().cpu().tolist()
            svob_true = svo_boundary_labels.detach().cpu().tolist()
            role_coarse_true = role_coarse_labels.detach().cpu().tolist()
            role_oblique_true = role_oblique_labels.detach().cpu().tolist()
            role_true    = role_labels.detach().cpu().tolist()
            voice_true   = voice_labels.detach().cpu().tolist()
            certainty_true = certainty_labels.detach().cpu().tolist()
            gender_true  = gender_labels.detach().cpu().tolist()
            number_true  = number_labels.detach().cpu().tolist()
            person_true  = person_labels.detach().cpu().tolist()
            gov_verb_true = gov_verb_labels.detach().cpu().tolist()
            n_all = len(boundary_labels)
            vfam_true      = verb_family_labels.detach().cpu().tolist()      if verb_family_labels is not None else [VERB_FAMILY_NONE_ID] * n_all
            vfam_fine_true = verb_family_fine_labels.detach().cpu().tolist() if verb_family_fine_labels is not None else [VERB_FAMILY_FINE_NONE_ID] * n_all
            vpol_true      = verb_polarity_labels.detach().cpu().tolist()    if verb_polarity_labels is not None else [VERB_POLARITY_NONE_ID] * n_all
            vasp_true      = verb_aspect_labels.detach().cpu().tolist()      if verb_aspect_labels is not None else [VERB_ASPECT_NONE_ID] * n_all
            vsrc_true      = verb_source_labels.detach().cpu().tolist()      if verb_source_labels is not None else [VERB_SOURCE_NONE_ID] * n_all


        # Accumulate boundary / coarse
        all_b_true.extend(b_true)
        all_b_pred.extend(b_pred)

        all_c_true.extend(c_true)
        all_c_pred.extend(c_pred)


        # Coarse metrics = POSITIVE ONLY (boundary=1)
        for bt, ct, cp in zip(b_true, c_true, c_pred):
            if bt == 1:
                all_c_true_pos.append(ct)
                all_c_pred_pos.append(cp)

        all_svob_true.extend(svob_true)
        all_svob_pred.extend(svob_pred)

        # Role coarse metrics = spans avec vrai rôle SVO (< ROLE_COARSE_NONE_ID, != OTHER)
        # OTHER est dans le softmax pour cascade inférence mais exclu du training + métriques
        for rct, rcp, rcfr in zip(role_coarse_true, role_coarse_pred_raw, role_coarse_from_role_pred_raw):
            if 0 <= rct < ROLE_COARSE_NONE_ID and rct != ROLE_COARSE_OTHER_ID:
                all_rc_true.append(rct)
                all_rc_pred.append(rcp)
                all_rc_from_role_true.append(rct)
                all_rc_from_role_pred.append(rcfr)

        # Role oblique metrics = spans COARSE-OBLIQ annotés (role_coarse_true==OBLIQ),
        # tous sous-types inclus (OBLIQUE_GENERIC id=0 inclus — cohérence avec la loss).
        # Conditionner sur role_coarse gold (pas sur role_oblique_true>0) évite le biais
        # d'évaluation qui masquait 77.7% des données d'obligues.
        _OBLIQ_RC = ROLE_COARSE2ID["OBLIQ"]
        for rot, rop, rct in zip(role_oblique_true, role_oblique_pred_raw, role_coarse_true):
            if rct == _OBLIQ_RC and rot >= 0 and rot < ROLE_OBLIQUE_NONE_ID:
                all_ro_true.append(rot)
                all_ro_pred.append(rop)

        # Role oblique CASCADE — gate = role_coarse_from_role prédit OBLIQ
        # Simule l'inférence réelle : un span reçoit role_oblique seulement si prédit OBLIQ par la dérivée
        for rot, rop, rcfr in zip(role_oblique_true, role_oblique_pred_raw, role_coarse_from_role_pred_raw):
            if rcfr == _OBLIQ_RC and rot >= 0 and rot < ROLE_OBLIQUE_NONE_ID:
                all_ro_cascaded_true.append(rot)
                all_ro_cascaded_pred.append(rop)

        # Role fin (12 labels) : spans avec rôle annoté (!= NONE = 6)
        # NOTE : NONE est à l'index 6 (pas en fin) → on ne peut pas utiliser < ROLE_NONE_ID
        # car cela exclurait les obliques étendues 7-11
        for rt, rp in zip(role_true, role_pred_raw):
            if rt >= 0 and rt != ROLE_NONE_ID:
                all_role_true.append(rt)
                all_role_pred.append(rp)

        # Fine metrics = POSITIVE ONLY
        for bt, ft, fp in zip(b_true, f_true, f_pred):
            if bt == 1:
                all_f_true_pos.append(ft)
                all_f_pred_pos.append(fp)

        # SVO role metrics = spans avec rôle annoté (role != NONE, inclut obliques étendus 7-11)
        for vt, vp in zip(voice_true, voice_pred_raw):
            if vt < NUM_VOICE:
                all_voice_true.append(vt)
                all_voice_pred.append(vp)
        # Certainty : spans avec certainty annoté (label != CERTAINTY_NONE_ID)
        for ct, cp in zip(certainty_true, certainty_pred_raw):
            if ct < NUM_CERTAINTY:
                all_certainty_true.append(ct)
                all_certainty_pred.append(cp)
        # Morpho : sur spans avec gender/number/person annotés
        for rt, gt, gp, nt, np_, pt, pp in zip(
            role_coarse_true, gender_true, gender_pred_raw,
            number_true, number_pred_raw,
            person_true, person_pred_raw
        ):
            if gt < NUM_GENDER:
                all_gender_true.append(gt)
                all_gender_pred.append(gp)
            if nt < NUM_NUMBER:
                all_number_true.append(nt)
                all_number_pred.append(np_)
            if pt < NUM_PERSON:
                all_person_true.append(pt)
                all_person_pred.append(pp)

        # verb pointer accuracy : spans avec gov_verb_labels >= 0
        seq_len_ptr = vptr_logits_cpu.size(1)
        for gvt, gvp in zip(gov_verb_true, ptr_pred_raw):
            if gvt >= 0 and gvt < seq_len_ptr:
                all_ptr_true.append(gvt)
                all_ptr_pred.append(gvp)

        # VerbFam : seulement les verb_trigger annotés (label < sentinel)
        for vft, vfp in zip(vfam_true, vfam_pred_raw):
            if vft < VERB_FAMILY_NONE_ID:
                all_vfam_true.append(vft)
                all_vfam_pred.append(vfp)
        for vft, vfp in zip(vfam_fine_true, vfam_fine_pred_raw):
            if vft < VERB_FAMILY_FINE_NONE_ID:
                all_vfam_fine_true.append(vft)
                all_vfam_fine_pred.append(vfp)
        for vpt, vpp in zip(vpol_true, vpol_pred_raw):
            if vpt < VERB_POLARITY_NONE_ID:
                all_vpol_true.append(vpt)
                all_vpol_pred.append(vpp)
        for vat, vap in zip(vasp_true, vasp_pred_raw):
            if vat < VERB_ASPECT_NONE_ID:
                all_vasp_true.append(vat)
                all_vasp_pred.append(vap)
        for vst, vsp in zip(vsrc_true, vsrc_pred_raw):
            if vst < VERB_SOURCE_NONE_ID:
                all_vsrc_true.append(vst)
                all_vsrc_pred.append(vsp)

        # ── Inline HN mining : collecter erreurs par candidat ────────────────
        if collect_hn:
            from collections import defaultdict
            pos_map = []
            for bi, sample_spans in enumerate(spans):
                for _ in range(len(sample_spans)):
                    pos_map.append(bi)

            n_global = len(pos_map)
            si_list = (
                span_indices.detach().cpu().tolist()
                if span_indices is not None
                else list(range(n_global))
            )

            raw_results: list = [None] * n_global
            for out_idx, in_idx in enumerate(si_list):
                bp = b_pred[out_idx]; bl = b_true[out_idx]
                cp = c_pred[out_idx]; cl = c_true[out_idx]
                fp_ = f_pred[out_idx]; fl = f_true[out_idx]

                pred_coarse = COARSE_LABELS[cp] if cp < len(COARSE_LABELS) else "?"
                pred_fine   = FINE_LABELS[fp_]  if fp_ < len(FINE_LABELS)  else "?"

                if bp != bl:
                    err = "FP_BOUNDARY" if (bl == 0 and bp == 1) else "FN_BOUNDARY"
                elif bl == 1 and cp != cl:
                    err = "COARSE_ERR"
                elif bl == 1 and fp_ != fl:
                    err = "FINE_ERR"
                else:
                    err = None

                # SVO boundary mining : FP/FN sur la tête verbe/pronom
                svob_p_i = svob_pred[out_idx]; svob_l_i = svob_true[out_idx]
                if svob_p_i != svob_l_i:
                    svo_b_err = "FP_SVO_BOUNDARY" if (svob_l_i == 0 and svob_p_i == 1) else "FN_SVO_BOUNDARY"
                else:
                    svo_b_err = None

                # Role coarse mining : erreur sur SUBJ/OBJ/OBLIQ annoté
                # → la cascade SVO→NER dépend de la qualité des rôles coarse
                rcp_i = role_coarse_pred_raw[out_idx]; rct_i = role_coarse_true[out_idx]
                if 0 <= rct_i < ROLE_COARSE_NONE_ID and rcp_i != rct_i:
                    role_coarse_err = "ROLE_COARSE_ERR"
                else:
                    role_coarse_err = None

                raw_results[in_idx] = (err, pred_coarse, pred_fine, svo_b_err, role_coarse_err)

            per_row: dict[int, list] = defaultdict(list)
            for in_idx, bi in enumerate(pos_map):
                per_row[bi].append(raw_results[in_idx])

            for bi, row_results in per_row.items():
                hn_results_by_id[batch["ids"][bi]] = row_results

    if train and (len(loader) % accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model)

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "boundary_f1": safe_macro_f1_local(all_b_true, all_b_pred),
        "coarse_macro_f1": safe_macro_f1_local(
            all_c_true_pos,
            all_c_pred_pos,
            labels=list(range(len(COARSE_LABELS) - 1))  # excl. NONE
        ),
        # Fine : macro sur classes PRÉSENTES seulement (v8.1 a retiré des labels rares
        # comme hint_measure/-1901, hint_object_generic/-321 → leur absence ferait F1=0
        # et diluerait le macro sur 38 classes)
        "fine_macro_f1": safe_macro_f1_local(
            all_f_true_pos,
            all_f_pred_pos,
            labels=[l for l in range(len(FINE_LABELS)) if l in set(all_f_true_pos)]
        ),
        # Fine split : CONCRETE (entités nommées prototypiques) vs ABSTRACT (génériques/notions)
        # Permet de diagnostiquer : le modèle maîtrise-t-il les NER classiques mais peine sur
        # les labels abstraits (doctrine, state, notion, group_role, loc_generic…) ?
        "fine_concrete_f1": safe_macro_f1_local(
            [l for l in all_f_true_pos if l in FINE_CONCRETE_IDS],
            [p for l, p in zip(all_f_true_pos, all_f_pred_pos) if l in FINE_CONCRETE_IDS],
            labels=[l for l in FINE_CONCRETE_IDS if l in set(all_f_true_pos)]
        ) if any(l in FINE_CONCRETE_IDS for l in all_f_true_pos) else 0.0,
        "fine_abstract_f1": safe_macro_f1_local(
            [l for l in all_f_true_pos if l in FINE_ABSTRACT_IDS],
            [p for l, p in zip(all_f_true_pos, all_f_pred_pos) if l in FINE_ABSTRACT_IDS],
            labels=[l for l in FINE_ABSTRACT_IDS if l in set(all_f_true_pos)]
        ) if any(l in FINE_ABSTRACT_IDS for l in all_f_true_pos) else 0.0,
        "svo_boundary_f1": safe_macro_f1_local(all_svob_true, all_svob_pred),
        "role_coarse_macro_f1": safe_macro_f1_local(
            all_rc_true, all_rc_pred,
            labels=[l for l in range(NUM_ROLE_COARSE) if l in set(all_rc_true)]
        ) if all_rc_true else 0.0,
        # coarse dérivée depuis role_head — même mask, diagnostic comparatif
        "role_coarse_from_role_macro_f1": safe_macro_f1_local(
            all_rc_from_role_true, all_rc_from_role_pred,
            labels=[l for l in range(4) if l in set(all_rc_from_role_true)]
        ) if all_rc_from_role_true else 0.0,
        "role_oblique_macro_f1": safe_macro_f1_local(
            all_ro_true, all_ro_pred,
            labels=[l for l in range(NUM_ROLE_OBLIQUE) if l in set(all_ro_true)]
        ) if all_ro_true else 0.0,
        # role_oblique CASCADE — gate = role_coarse_from_role prédit OBLIQ (mode inférence réelle)
        # Mesure la précision de role_oblique_head sur les spans correctement gatés par la dérivée
        "role_oblique_cascaded_macro_f1": safe_macro_f1_local(
            all_ro_cascaded_true, all_ro_cascaded_pred,
            labels=[l for l in range(NUM_ROLE_OBLIQUE) if l in set(all_ro_cascaded_true)]
        ) if all_ro_cascaded_true else 0.0,
        # role fin (12 labels) : spans avec rôle annoté (< ROLE_NONE_ID)
        "role_macro_f1": safe_macro_f1_local(
            all_role_true, all_role_pred,
            labels=[l for l in range(NUM_ROLE) if l in set(all_role_true)]
        ) if all_role_true else 0.0,
        "voice_macro_f1": safe_macro_f1_local(all_voice_true, all_voice_pred) if all_voice_true else 0.0,
        "certainty_macro_f1": safe_macro_f1_local(
            all_certainty_true, all_certainty_pred,
            labels=[l for l in range(NUM_CERTAINTY) if l in set(all_certainty_true)]
        ) if all_certainty_true else 0.0,
        # Morpho : seulement les classes PRÉSENTES dans les vrais labels
        # (ex: N/neutre n'existe jamais dans les données → l'inclure donnerait F1_N=0
        #  et diluerait le macro de 1/3)
        "gender_macro_f1": safe_macro_f1_local(
            all_gender_true, all_gender_pred,
            labels=[l for l in range(NUM_GENDER) if l in set(all_gender_true)]
        ) if all_gender_true else 0.0,
        "number_macro_f1": safe_macro_f1_local(
            all_number_true, all_number_pred,
            labels=[l for l in range(NUM_NUMBER) if l in set(all_number_true)]
        ) if all_number_true else 0.0,
        "person_macro_f1": safe_macro_f1_local(
            all_person_true, all_person_pred,
            labels=[l for l in range(NUM_PERSON) if l in set(all_person_true)]
        ) if all_person_true else 0.0,
        # verb pointer : accuracy (exact token match)
        "verb_ptr_acc": (
            sum(t == p for t, p in zip(all_ptr_true, all_ptr_pred)) / len(all_ptr_true)
            if all_ptr_true else 0.0
        ),
        "verb_ptr_n": len(all_ptr_true),
        # VerbFam heads (verb_trigger uniquement)
        "verb_family_macro_f1": safe_macro_f1_local(
            all_vfam_true, all_vfam_pred,
            labels=[l for l in range(NUM_VERB_FAMILY) if l in set(all_vfam_true)]
        ) if all_vfam_true else 0.0,
        "verb_family_fine_macro_f1": safe_macro_f1_local(
            all_vfam_fine_true, all_vfam_fine_pred,
            labels=[l for l in range(NUM_VERB_FAMILY_FINE) if l in set(all_vfam_fine_true)]
        ) if all_vfam_fine_true else 0.0,
        "verb_polarity_macro_f1": safe_macro_f1_local(
            all_vpol_true, all_vpol_pred,
            labels=[l for l in range(NUM_VERB_POLARITY) if l in set(all_vpol_true)]
        ) if all_vpol_true else 0.0,
        "verb_aspect_macro_f1": safe_macro_f1_local(
            all_vasp_true, all_vasp_pred,
            labels=[l for l in range(NUM_VERB_ASPECT) if l in set(all_vasp_true)]
        ) if all_vasp_true else 0.0,
        "verb_source_macro_f1": safe_macro_f1_local(
            all_vsrc_true, all_vsrc_pred,
            labels=[l for l in range(NUM_VERB_SOURCE) if l in set(all_vsrc_true)]
        ) if all_vsrc_true else 0.0,
        "boundary_report": classification_report(
            all_b_true, all_b_pred, digits=3, zero_division=0
        ) if all_b_true else "N/A",
        "coarse_report": classification_report(
            all_c_true_pos, all_c_pred_pos,
            labels=list(range(len(COARSE_LABELS) - 1)),
            target_names=COARSE_LABELS[:-1], digits=3, zero_division=0
        ) if all_c_true_pos else "N/A",
        "fine_report": classification_report(
            all_f_true_pos, all_f_pred_pos,
            labels=list(range(len(FINE_LABELS))),
            target_names=FINE_LABELS, digits=3, zero_division=0
        ) if all_f_true_pos else "N/A",
        # svo_boundary : tête binaire VERB / non-VERB (verb_trigger + pron vs tout le reste)
        "svo_boundary_report": classification_report(
            all_svob_true, all_svob_pred,
            labels=[0, 1], target_names=["non_verb", "verb_trigger"],
            digits=3, zero_division=0
        ) if all_svob_true else "N/A",
        # role_coarse : SUBJ/OBJ/OBLIQ/APPOS
        "role_coarse_report": classification_report(
            all_rc_true, all_rc_pred,
            labels=[l for l in range(NUM_ROLE_COARSE) if l != ROLE_COARSE_NONE_ID and l in set(all_rc_true)],
            target_names=[ROLE_COARSE_LABELS[l] for l in range(NUM_ROLE_COARSE) if l != ROLE_COARSE_NONE_ID and l in set(all_rc_true)],
            digits=3, zero_division=0
        ) if all_rc_true else "N/A",
        # role_oblique : sous-types OBLIQUE (NONE exclu)
        "role_oblique_report": classification_report(
            all_ro_true, all_ro_pred,
            labels=[l for l in range(NUM_ROLE_OBLIQUE) if l != ROLE_OBLIQUE_NONE_ID and l in set(all_ro_true)],
            target_names=[ROLE_OBLIQUE_LABELS[l] for l in range(NUM_ROLE_OBLIQUE) if l != ROLE_OBLIQUE_NONE_ID and l in set(all_ro_true)],
            digits=3, zero_division=0
        ) if all_ro_true else "N/A",
        # role fin (12 labels) : SUBJECT/OBJECT/OBLIQUE_* (NONE exclu)
        "role_report": classification_report(
            all_role_true, all_role_pred,
            labels=[l for l in range(NUM_ROLE) if l != ROLE_NONE_ID and l in set(all_role_true)],
            target_names=[ROLE_LABELS[l] for l in range(NUM_ROLE) if l != ROLE_NONE_ID and l in set(all_role_true)],
            digits=3, zero_division=0
        ) if all_role_true else "N/A",
        # VerbFam reports
        "verb_family_report": classification_report(
            all_vfam_true, all_vfam_pred,
            labels=[l for l in range(NUM_VERB_FAMILY) if l in set(all_vfam_true)],
            target_names=[VERB_FAMILY_LABELS[l] for l in range(NUM_VERB_FAMILY) if l in set(all_vfam_true)],
            digits=3, zero_division=0
        ) if all_vfam_true else "N/A",
        "verb_family_fine_report": classification_report(
            all_vfam_fine_true, all_vfam_fine_pred,
            labels=[l for l in range(NUM_VERB_FAMILY_FINE) if l in set(all_vfam_fine_true)],
            target_names=[VERB_FAMILY_FINE_LABELS[l] for l in range(NUM_VERB_FAMILY_FINE) if l in set(all_vfam_fine_true)],
            digits=3, zero_division=0
        ) if all_vfam_fine_true else "N/A",
        "verb_polarity_report": classification_report(
            all_vpol_true, all_vpol_pred,
            labels=[l for l in range(NUM_VERB_POLARITY) if l in set(all_vpol_true)],
            target_names=[VERB_POLARITY_LABELS[l] for l in range(NUM_VERB_POLARITY) if l in set(all_vpol_true)],
            digits=3, zero_division=0
        ) if all_vpol_true else "N/A",
        "verb_aspect_report": classification_report(
            all_vasp_true, all_vasp_pred,
            labels=[l for l in range(NUM_VERB_ASPECT) if l in set(all_vasp_true)],
            target_names=[VERB_ASPECT_LABELS[l] for l in range(NUM_VERB_ASPECT) if l in set(all_vasp_true)],
            digits=3, zero_division=0
        ) if all_vasp_true else "N/A",
        "verb_source_report": classification_report(
            all_vsrc_true, all_vsrc_pred,
            labels=[l for l in range(NUM_VERB_SOURCE) if l in set(all_vsrc_true)],
            target_names=[VERB_SOURCE_LABELS[l] for l in range(NUM_VERB_SOURCE) if l in set(all_vsrc_true)],
            digits=3, zero_division=0
        ) if all_vsrc_true else "N/A",
        "hn_results_by_id": hn_results_by_id,
    }

    if not train:
        metrics.update(build_fine_diagnostics(all_f_true_pos, all_f_pred_pos, split_name=eval_split))

    return metrics


def apply_inline_hn(
    train_ds: "MultiTaskSpanDataset",
    results_by_id: dict,
    boosts: dict,
    decay: float,
    max_weight: float,
    min_weight: float,
) -> dict:
    """
    Met à jour in-memory les sample_weights de train_ds.rows
    à partir des erreurs collectées pendant l'epoch de training.
    Retourne un dict de stats pour le logging.
    """
    from collections import Counter
    stats: Counter = Counter()

    # Construire un index id → row_index pour un accès rapide
    id_to_idx = {row["id"]: i for i, row in enumerate(train_ds.rows)}

    for rid, row_results in results_by_id.items():
        idx = id_to_idx.get(rid)
        if idx is None:
            continue
        row = train_ds.rows[idx]
        valid_cands = [c for c in row["candidates"] if _hn_is_valid(c)]

        for i, c in enumerate(valid_cands):
            if i >= len(row_results):
                break
            entry = row_results[i]
            if entry is None:
                w = c.get("sample_weight", 1.0)
                c["sample_weight"] = max(min_weight, 1.0 + (w - 1.0) * decay)
                stats["decayed"] += 1
                continue

            err, pred_coarse, pred_fine, svo_b_err, role_coarse_err = entry if len(entry) == 5 else (*entry, None)
            if err is None and svo_b_err is None and role_coarse_err is None:
                w = c.get("sample_weight", 1.0)
                c["sample_weight"] = max(min_weight, 1.0 + (w - 1.0) * decay)
                stats["decayed"] += 1
            else:
                new_w = c.get("sample_weight", 1.0)
                if err is not None:
                    base = boosts.get(err, 2.0)
                    if err == "FP_BOUNDARY" and pred_coarse in _LOW_PRECISION_COARSE:
                        base *= _FP_LOW_PREC_EXTRA
                    if err == "FINE_ERR" and pred_fine in _LOW_F1_FINE:
                        base *= _FINE_ERR_EXTRA
                    new_w *= base
                    c["neg_type"] = err
                    stats[err] += 1
                if svo_b_err is not None:
                    svo_base = boosts.get(svo_b_err, 2.0)
                    new_w *= svo_base
                    if "neg_type" not in c or c.get("neg_type") == "unknown":
                        c["neg_type"] = svo_b_err
                    stats[svo_b_err] += 1
                if role_coarse_err is not None:
                    rc_base = boosts.get(role_coarse_err, 1.5)
                    new_w *= rc_base
                    if "neg_type" not in c or c.get("neg_type") == "unknown":
                        c["neg_type"] = role_coarse_err
                    stats[role_coarse_err] += 1
                c["sample_weight"] = min(max_weight, new_w)

    return dict(stats)


def _hn_is_valid(c: dict) -> bool:
    ts, te = c.get("tok_start"), c.get("tok_end")
    return isinstance(ts, int) and isinstance(te, int) and ts >= 0 and te >= ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)

    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=30,
                        help="Nombre maximum d'epochs (défaut=30 ; l'early stopping arrêtera avant si patience atteinte)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--lambda-boundary", type=float, default=2.5)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--lambda-fine", type=float, default=1.8)
    parser.add_argument("--lambda-svo", type=float, default=0.6,
                        help="Pondération de la loss SVO (défaut=0.8)")
    parser.add_argument("--lambda-role-coarse", type=float, default=0.1,
                        help="Pondération de la loss rôle SVO coarse SUBJ/OBJ/OBLIQ/OTHER (défaut=0.1)")
    parser.add_argument("--lambda-role-oblique", type=float, default=0.15,
                        help="Pondération de la loss rôle oblique fin 10 sous-types (défaut=0.15)")
    parser.add_argument("--lambda-role", type=float, default=0.0,
                        help="Ancienne tête rôle unifiée 12 labels (défaut=0 = désactivée)")
    parser.add_argument("--lambda-voice", type=float, default=0.15,
                        help="Pondération de la loss voice ACTIVE/PASSIVE (défaut=0.5)")
    parser.add_argument("--lambda-certainty", type=float, default=0.4,
                        help="Pondération de la loss certainty (défaut=0.4)")
    parser.add_argument("--lambda-svo-boundary", type=float, default=0.7,
                        help="Pondération de la loss svo_boundary (détection verbes/pronom, défaut=0.9)")
    parser.add_argument("--lambda-morpho", type=float, default=0.3,
                        help="Pondération de la loss morpho gender+number+person (défaut=0.3)")
    parser.add_argument("--lambda-verb-ptr", type=float, default=0.5,
                        help="Pondération de la loss verb-pointer arg→verb (défaut=0.5)")
    parser.add_argument("--lambda-compat", type=float, default=0.2,
                        help="Pondération loss compat inter-têtes")
    parser.add_argument("--lambda-verb-family",      type=float, default=0.0)
    parser.add_argument("--lambda-verb-family-fine", type=float, default=0.0)
    parser.add_argument("--lambda-verb-polarity",    type=float, default=0.0)
    parser.add_argument("--lambda-verb-aspect",      type=float, default=0.0)
    parser.add_argument("--lambda-verb-source",      type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma pour boundary (0=CE, 2.0=focal)")
    parser.add_argument("--focal-fine-gamma", type=float, default=0.0,
                        help="Focal loss gamma pour tête fine (0=CE, 1.5=recommandé pour classes rares)")
    parser.add_argument("--focal-coarse-gamma", type=float, default=0.0,
                        help="Focal loss gamma sur coarse positifs uniquement (0=CE, 1.0=recommandé OBJECT/EVENT)")
    parser.add_argument("--focal-role-gamma", type=float, default=0.0,
                        help="Focal loss gamma sur tête SVO role (0=CE, 2.0=recommandé OBLIQUE_*/APPOS rares). "
                             "Concentre l'apprentissage sur les rôles difficiles sans amplifier les gradients "
                             "des classes rares (contrairement aux class weights).")
    parser.add_argument("--head-lr-multiplier", type=float, default=5.0,
                        help="Multiplicateur LR pour les heads vs encoder")
    parser.add_argument("--layer-lr-decay", type=float, default=0.9,
                        help="Decay LR par couche (1.0=désactivé, 0.9=recommandé)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Decay EMA (0.0=désactivé, 0.999=recommandé)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile(model, dynamic=True) — +20-40%% sur H100/A100 (PyTorch 2.0+). "
                             "dynamic=True gère les shapes variables de spans entre batches.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Active gradient checkpointing sur l'encodeur — réduit VRAM activations ~30%% "
                             "(recompute forward partiel pendant backward). Permet BS plus élevé sur H100.")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Nombre d'epochs de linear warmup LR")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping : nombre d'epochs sans amélioration avant arrêt (0=désactivé)")
    parser.add_argument("--min-delta", type=float, default=1e-3,
                        help="Early stopping : amélioration minimale du score pour réinitialiser le compteur (défaut=0.001)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping max norm (0.0=désactivé, 1.0=recommandé pour DeBERTa)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing pour coarse/fine CE")

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--amp", action="store_true",
                        help="Active Automatic Mixed Precision (BF16 sur CUDA). Gain typique: 2x vitesse.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Nombre de workers DataLoader (défaut=4, 0=mono-thread).")
    parser.add_argument("--class-weights", choices=["none", "auto"], default="auto")
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Puissance appliquée aux poids inverse-fréquence. 1.0 = brut, 0.5 = tempéré (recommandé)."
    )
    parser.add_argument(
        "--min-coarse-none-weight",
        type=float,
        default=0.05,
        help="Poids minimum autorisé pour la classe NONE de la tête coarse."
    )
    parser.add_argument(
        "--ignore-coarse-none",
        action="store_true",
        default=False,
        help="Si activé, exclut les spans NONE de la loss coarse (positive-only coarse). "
             "La tête boundary gère déjà entité/non-entité → signal redondant supprimé."
    )
    parser.add_argument(
        "--min-fine-none-weight",
        type=float,
        default=0.05,
        help="Poids minimum autorisé pour la classe NONE de la tête fine."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint pour reprendre le training."
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Force l'epoch de départ quand on reprend depuis un checkpoint."
    )

    # ── Inline Hard Negative Mining ──────────────────────────────────────────
    parser.add_argument("--hn-every",        type=int,   default=0,
                        help="Appliquer le HN mining toutes les N epochs (0=désactivé)")
    parser.add_argument("--hn-boost-fp",     type=float, default=3.5,
                        help="Boost FP_BOUNDARY (défaut=3.5)")
    parser.add_argument("--hn-boost-fn",     type=float, default=2.0,
                        help="Boost FN_BOUNDARY (défaut=2.0)")
    parser.add_argument("--hn-boost-coarse", type=float, default=2.5,
                        help="Boost COARSE_ERR (défaut=2.5)")
    parser.add_argument("--hn-boost-fine",   type=float, default=2.0,
                        help="Boost FINE_ERR (défaut=2.0)")
    parser.add_argument("--hn-boost-fp-svo",  type=float, default=3.0,
                        help="Boost FP_SVO_BOUNDARY : span prédit verbe/pronom mais pas gold (défaut=3.0)")
    parser.add_argument("--hn-boost-fn-svo",  type=float, default=2.0,
                        help="Boost FN_SVO_BOUNDARY : span gold verbe/pronom non détecté (défaut=2.0)")
    parser.add_argument("--hn-boost-role-coarse", type=float, default=1.5,
                        help="Boost ROLE_COARSE_ERR : erreur SUBJ/OBJ/OBLIQ — modéré car signal cascade SVO→NER (défaut=1.5)")
    parser.add_argument("--hn-decay",        type=float, default=0.85,
                        help="Décroissance des poids bien prédits (défaut=0.85)")
    parser.add_argument("--hn-max-weight",   type=float, default=8.0,
                        help="Poids maximum (défaut=8.0)")
    parser.add_argument("--hn-min-weight",   type=float, default=0.3,
                        help="Poids minimum (défaut=0.3)")

    # ── Dynamic Loss Weighting ──────────────────────────────────────────
    parser.add_argument("--loss-weighting", choices=["fixed", "uncertainty", "gradnorm"],
                        default="fixed",
                        help="Stratégie de pondération des losses multi-task. "
                             "fixed=lambdas fixes (défaut), "
                             "uncertainty=Kendall 2018 (learnable sigma par tête), "
                             "gradnorm=Chen ICML 2018 (gradient norm balancing)")
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5,
                        help="GradNorm alpha (asymétrie du balancement, 1.5=recommandé)")
    parser.add_argument("--gradnorm-every", type=int, default=10,
                        help="GradNorm update every N optimizer steps (10=recommandé pour limiter le coût)")

    # ── W&B ──────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb-project", type=str, default="pimpmyrag-ner",
                        help="Nom du projet W&B (défaut: pimpmyrag-ner). Mettre '' pour désactiver.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Nom du run W&B (défaut: autogénéré)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="ID d'un run W&B existant à reprendre (pour continuer le même run entre epochs)")
    parser.add_argument("--wandb-id-file", type=str, default="wandb_run_id.txt",
                        help="Fichier où sauvegarder/lire le run ID W&B entre les epochs")
    parser.add_argument("--wandb-tags", type=str, default="",
                        help="Tags W&B séparés par des virgules (ex: 'v6.3,deberta,5090')")
    parser.add_argument("--ner-only-score", action="store_true",
                        help="Calcule le score d'early stopping uniquement sur les têtes NER (sans SVO)")

    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ device = {device}")

    # ── W&B init ─────────────────────────────────────────────────────────────
    _wandb_enabled = False
    if _WANDB_AVAILABLE and args.wandb_project:
        try:
            # Résolution du run ID : argument CLI > fichier persisté > nouveau run
            wandb_run_id = args.wandb_run_id
            if not wandb_run_id and args.wandb_id_file:
                try:
                    with open(args.wandb_id_file) as f:
                        wandb_run_id = f.read().strip() or None
                except FileNotFoundError:
                    pass

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                id=wandb_run_id,
                resume="allow",
                tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
                config={
                    # Hyperparams training
                    "model_name":        args.model_name,
                    "lr":                args.lr,
                    "head_lr_multiplier": args.head_lr_multiplier,
                    "batch_size":        args.batch_size,
                    "accum_steps":       args.accum_steps,
                    "max_epochs":        args.epochs,
                    "warmup_epochs":     args.warmup_epochs,
                    "layer_lr_decay":    args.layer_lr_decay,
                    "ema_decay":         args.ema_decay,
                    "focal_gamma":       args.focal_gamma,
                    "max_grad_norm":     args.max_grad_norm,
                    # Lambdas
                    "lambda_boundary":   args.lambda_boundary,
                    "lambda_coarse":     args.lambda_coarse,
                    "lambda_fine":       args.lambda_fine,
                    "lambda_svo_boundary": args.lambda_svo_boundary,
                    "lambda_svo":        args.lambda_svo,
                    "lambda_role_coarse": args.lambda_role_coarse,
                    "lambda_role_oblique": args.lambda_role_oblique,
                    "lambda_role":       args.lambda_role,
                    "lambda_voice":      args.lambda_voice,
                    "lambda_certainty":  args.lambda_certainty,
                    "lambda_morpho":     args.lambda_morpho,
                    "lambda_verb_ptr":          args.lambda_verb_ptr,
                    "lambda_compat":            args.lambda_compat,
                    "lambda_verb_family":       args.lambda_verb_family,
                    "lambda_verb_family_fine":  args.lambda_verb_family_fine,
                    "lambda_verb_polarity":     args.lambda_verb_polarity,
                    "lambda_verb_aspect":       args.lambda_verb_aspect,
                    "lambda_verb_source":       args.lambda_verb_source,
                    # Schema
                    "num_fine":          len(FINE_LABELS),
                    "num_coarse":        len(COARSE_LABELS),
                    # Dataset
                    "train_path":        args.train,
                    "val_path":          args.val,
                },
            )
            # Persiste le run ID pour les epochs suivantes
            if args.wandb_id_file:
                with open(args.wandb_id_file, "w") as f:
                    f.write(wandb.run.id)
            _wandb_enabled = True
            print(f"📊 W&B run: {wandb.run.name} | {wandb.run.url}")
        except Exception as e:
            print(f"⚠️  W&B init échoué ({e}) — training sans W&B")
    else:
        if not _WANDB_AVAILABLE:
            print("⚠️  wandb non installé — pip install wandb pour activer le tracking")
        elif not args.wandb_project:
            print("ℹ️  W&B désactivé (--wandb-project vide)")


    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    train_ds = MultiTaskSpanDataset(args.train, tokenizer, max_length=args.max_length)
    val_ds = MultiTaskSpanDataset(args.val, tokenizer, max_length=args.max_length)
    test_ds = MultiTaskSpanDataset(args.test, tokenizer, max_length=args.max_length)

    collate_fn = make_collate_fn(tokenizer)

    pin_memory = (device == "cuda")
    num_workers = args.num_workers
    # persistent_workers évite de re-forker les workers à chaque epoch
    persistent = (num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    print(f"📦 DataLoaders: num_workers={num_workers}, pin_memory={pin_memory}, persistent={persistent}")

    model = SpanMultiTaskModel(model_name=args.model_name, num_coarse=len(COARSE_LABELS)).to(device).float()

    # Gradient checkpointing — réduit VRAM activations encodeur ~30% (recompute pendant backward)
    # Permet d'aller BS=192→320 sur H100 sans OOM. Overhead backward ~+15% acceptable.
    if getattr(args, "gradient_checkpointing", False):
        model.encoder.gradient_checkpointing_enable()
        print("🔖 Gradient checkpointing activé sur l'encodeur (économie VRAM ~30%)")

    total_epochs = args.epochs

    # torch.compile — +20-40% sur H100/A100 (PyTorch 2.0+, CUDA 12+)
    # dynamic=True : gère les shapes variables (candidats spans différents par batch)
    # mode="reduce-overhead" : réduit le overhead PyTorch sans fullgraph (plus stable)
    # ⚠️  DeBERTa-v3 INCOMPATIBLE avec torch.compile :
    #     - attention disentangled utilise des opérations CUDA custom non-traçables
    #     - mode="reduce-overhead" (CUDA graphs) exige des shapes statiques → crash au 1er forward
    #     - compile de DeBERTa dépasse 15min et ne converge pas
    _is_deberta = "deberta" in args.model_name.lower()
    if getattr(args, "compile", False) and device == "cuda":
        if _is_deberta:
            print(
                f"⚠️  torch.compile ignoré pour DeBERTa ({args.model_name}) — "
                "attention disentangled incompatible avec CUDA graphs (mode=reduce-overhead). "
                "Le gain BF16 + batch size élevé suffit sur H100/A100."
            )
        else:
            try:
                model = torch.compile(model, dynamic=True, mode="reduce-overhead")
                print("⚡ torch.compile activé (dynamic=True, mode=reduce-overhead)")
            except Exception as e:
                print(f"⚠️  torch.compile échoué ({e}) — run sans compile")

    # AMP BF16 — pas de GradScaler nécessaire (BF16 garde la même dynamique que FP32)
    # Supporté nativement sur Ampere+ (RTX 3090/4090/5090, A100…)
    use_amp = args.amp and (device == "cuda")
    scaler = None   # GradScaler inutile en BF16
    if use_amp:
        print("⚡ AMP (BF16) activé — pas de GradScaler (stable pour DeBERTa)")
    else:
        print("🔢 FP32 (AMP désactivé)")


    # Differential LR avec layer-wise decay
    head_lr = args.lr * args.head_lr_multiplier
    param_groups = get_layerwise_param_groups(model, args.lr, head_lr, decay=args.layer_lr_decay)
    optimizer = AdamW(param_groups)

    # ── Dynamic loss weighting ─────────────────────────────────────────────
    initial_lambdas = {
        "boundary": args.lambda_boundary, "coarse": args.lambda_coarse,
        "fine": args.lambda_fine, "svo_boundary": args.lambda_svo_boundary,
        "svo": args.lambda_svo,
        "voice": args.lambda_voice, "certainty": args.lambda_certainty,
        "morpho": args.lambda_morpho, "verb_ptr": args.lambda_verb_ptr,
        "compat": args.lambda_compat,
    }
    weighting = create_weighting(args.loss_weighting, initial_lambdas,
                                 alpha=args.gradnorm_alpha)
    weighting = weighting.to(device)
    if not isinstance(weighting, FixedWeighting):
        # Add weighting parameters to optimizer with separate LR
        wt_lr = 0.01 if args.loss_weighting == "uncertainty" else 0.025
        optimizer.add_param_group({
            "params": list(weighting.parameters()),
            "lr": wt_lr,
            "weight_decay": 0.0,
            "name": "loss_weighting",
        })
        print(f"🎛️  Loss weighting: {args.loss_weighting} (lr={wt_lr})")
    else:
        print(f"🎛️  Loss weighting: fixed (lambdas manuels)")

    # Log LR par couche
    print(f"📐 Layer-wise LR decay={args.layer_lr_decay}")
    for g in param_groups:
        print(f"   {g.get('name', '?'):<20} lr={g['lr']:.2e}")
    print(f"📐 Focal gamma: {args.focal_gamma}")

    boundary_w = coarse_w = fine_w = certainty_w = oblique_w = None
    verb_family_w = verb_polarity_w = verb_aspect_w = verb_source_w = None
    if args.class_weights == "auto":
        (
            boundary_w,
            coarse_w,
            fine_w,
            certainty_w,
            oblique_w,
            role_coarse_w,
            verb_family_w,
            verb_polarity_w,
            verb_aspect_w,
            verb_source_w,
            boundary_counts,
            coarse_counts,
            fine_counts,
            certainty_counts,
            oblique_counts,
            role_coarse_counts,
        ) = compute_class_weights_from_multitask_jsonl(
            args.train,
            power=args.class_weight_power,
        )

        coarse_none_idx = len(COARSE_LABELS) - 1
        fine_none_idx = len(FINE_LABELS) - 1

        coarse_w = apply_class_weight_floor(
            coarse_w,
            coarse_none_idx,
            args.min_coarse_none_weight,
        )
        fine_w = apply_class_weight_floor(
            fine_w,
            fine_none_idx,
            args.min_fine_none_weight,
        )

        print("⚖️ class weights auto activés")
        print(f"   class_weight_power       = {args.class_weight_power}")
        print(f"   min_coarse_none_weight   = {args.min_coarse_none_weight}")
        print(f"   ignore_coarse_none       = {args.ignore_coarse_none}")
        print(f"   min_fine_none_weight     = {args.min_fine_none_weight}")

        print("\n[boundary counts / weights]")
        for i in range(2):
            print(f"  class {i}: count={boundary_counts.get(i, 0)} weight={boundary_w[i].item():.6f}")

        print("\n[coarse counts / weights]")
        for i, name in enumerate(COARSE_LABELS):
            print(f"  {name:<10} count={coarse_counts.get(i, 0):>8} weight={coarse_w[i].item():.6f}")

        print("\n[fine counts / weights]")
        for i, name in enumerate(FINE_LABELS):
            print(f"  {name:<22} count={fine_counts.get(i, 0):>8} weight={fine_w[i].item():.6f}")

        print("\n[certainty counts / weights]")
        for i, name in enumerate(CERTAINTY_LABELS):
            print(f"  {name:<15} count={certainty_counts.get(i, 0):>8} weight={certainty_w[i].item():.6f}")

        print("\n[oblique fine counts / weights]  (CWP compense rareté ADVERSARY/SOURCE...)")
        for i, name in enumerate(ROLE_OBLIQUE_LABELS):
            print(f"  {name:<25} count={oblique_counts.get(i, 0):>6} weight={oblique_w[i].item():.6f}")


    else:
        print("⚖️ class weights désactivés")
        boundary_w = None
        coarse_w = None
        fine_w = None
        certainty_w = None
        oblique_w = None
        role_coarse_w = None

    best_score = -1.0
    start_epoch = 1
    epochs_no_improve = 0   # compteur early stopping

    # Boosts HN inline
    hn_boosts = {
        "FP_BOUNDARY":     args.hn_boost_fp,
        "FN_BOUNDARY":     args.hn_boost_fn,
        "COARSE_ERR":      args.hn_boost_coarse,
        "FINE_ERR":        args.hn_boost_fine,
        "FP_SVO_BOUNDARY": args.hn_boost_fp_svo,
        "FN_SVO_BOUNDARY": args.hn_boost_fn_svo,
        "ROLE_COARSE_ERR": args.hn_boost_role_coarse,
    }
    use_inline_hn = args.hn_every > 0
    if use_inline_hn:
        print(f"🎯 Inline HN mining activé toutes les {args.hn_every} epoch(s)")
        print(f"   boosts = {hn_boosts}")
        print(f"   decay={args.hn_decay}, max_weight={args.hn_max_weight}")

    # LR scheduler basé sur les epochs
    # - warmup_epochs_count>0 : LinearLR (LR/10 → LR plein) sur N epochs, puis CosineAnnealingLR.
    # - warmup_epochs_count=0 : CosineAnnealingLR directement (LR plein dès le départ).
    # NOTE : le pattern LinearLR(total_iters=0) + SequentialLR(milestones=[0]) appliquait LR×0.1
    # à TOUTES les epochs (nouveau process à chaque epoch) → fine_f1→0 quand NER_WARMUP actif.
    # La garde ci-dessous évite ce comportement pathologique.
    warmup_epochs_count = min(args.warmup_epochs, total_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs_count))
    if warmup_epochs_count > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_count)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs_count])
    else:
        scheduler = cosine_scheduler

    # EMA
    use_ema = args.ema_decay > 0.0
    ema = ModelEMA(model, decay=args.ema_decay) if use_ema else None
    if use_ema:
        print(f"📐 EMA activé (decay={args.ema_decay})")

    # Reprise éventuelle depuis checkpoint
    if args.resume is not None:
        if not os.path.exists(args.resume):
            print(f"⚠️  Checkpoint '{args.resume}' introuvable — démarrage à froid (pas de reprise)")
            args.resume = None
        else:
            print(f"⤴️ Reprise depuis checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            if "optim_state" in ckpt and ckpt["optim_state"] is not None:
                try:
                    # Sauvegarder les LR initiaux (définis dans param_groups par le script)
                    # AVANT le load_state_dict qui peut restaurer des LR=0 (fin de CosineAnneal)
                    initial_lrs = [pg['lr'] for pg in optimizer.param_groups]
                    optimizer.load_state_dict(ckpt["optim_state"])
                    # Réinitialiser les LR aux valeurs voulues par le script (pas le LR du checkpoint)
                    # Cela garantit un LR correct même si le checkpoint a LR=0 après scheduler.step()
                    for pg, lr in zip(optimizer.param_groups, initial_lrs):
                        pg['lr'] = lr
                    print(f"   LR restaurés : {[f'{lr:.2e}' for lr in initial_lrs]}")
                except Exception as e:
                    print(f"⚠️ Impossible de recharger l'optimizer state: {e}")

            if use_ema and "ema_state" in ckpt and ckpt["ema_state"] is not None:
                ema.shadow = {k: v.clone() for k, v in ckpt["ema_state"].items()}
                print("📐 EMA state rechargé depuis checkpoint")

            if not isinstance(weighting, FixedWeighting) and "weighting_state" in ckpt and ckpt["weighting_state"] is not None:
                try:
                    weighting.load_state_dict(ckpt["weighting_state"])
                    print(f"🎛️  Loss weighting state rechargé depuis checkpoint")
                except Exception as e:
                    print(f"⚠️ Impossible de recharger le weighting state: {e}")

            if args.start_epoch is not None:
                start_epoch = args.start_epoch
            else:
                start_epoch = int(ckpt.get("epoch", 0)) + 1

            best_score = ckpt.get("best_score", -1.0)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)

            print(
                f"✅ checkpoint rechargé | start_epoch={start_epoch} | best_score={best_score:.4f} | "
                f"epochs_no_improve={epochs_no_improve}"
            )

    for epoch in range(start_epoch, args.epochs + 1):
        # Log LR
        head_lrs = [pg['lr'] for pg in optimizer.param_groups if pg.get('name') == 'heads']
        enc_lrs = [pg['lr'] for pg in optimizer.param_groups if pg.get('name', '').startswith('layer_')]
        top_lr = enc_lrs[-1] if enc_lrs else args.lr
        print(f"\n🔧 Epoch {epoch} | LR top_layer={top_lr:.2e}, heads={head_lrs[0] if head_lrs else '?':.2e}")

        train_metrics = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            train=True,
            boundary_class_weights=boundary_w,
            coarse_class_weights=coarse_w,
            fine_class_weights=fine_w,
            certainty_class_weights=certainty_w,
            oblique_class_weights=oblique_w,
            role_coarse_class_weights=role_coarse_w,
            verb_family_class_weights=verb_family_w,
            verb_polarity_class_weights=verb_polarity_w,
            verb_aspect_class_weights=verb_aspect_w,
            verb_source_class_weights=verb_source_w,
            lambda_boundary=args.lambda_boundary,
            lambda_coarse=args.lambda_coarse,
            lambda_fine=args.lambda_fine,
            lambda_svo_boundary=args.lambda_svo_boundary,
            lambda_svo=args.lambda_svo,
            lambda_role_coarse=args.lambda_role_coarse,
            lambda_role_oblique=args.lambda_role_oblique,
            lambda_role=args.lambda_role,
            lambda_voice=args.lambda_voice,
            lambda_certainty=args.lambda_certainty,
            lambda_morpho=args.lambda_morpho,
            lambda_verb_ptr=args.lambda_verb_ptr,
            lambda_compat=args.lambda_compat,
            lambda_verb_family=args.lambda_verb_family,
            lambda_verb_family_fine=args.lambda_verb_family_fine,
            lambda_verb_polarity=args.lambda_verb_polarity,
            lambda_verb_aspect=args.lambda_verb_aspect,
            lambda_verb_source=args.lambda_verb_source,
            accum_steps=args.accum_steps,
            log_every=args.log_every,
            focal_gamma=args.focal_gamma,
            max_grad_norm=args.max_grad_norm,
            ema=ema,
            collect_hn=use_inline_hn and (epoch % args.hn_every == 0),
            scaler=scaler,
            focal_fine_gamma=args.focal_fine_gamma,
            focal_coarse_gamma=args.focal_coarse_gamma,
            focal_role_gamma=args.focal_role_gamma,
            ignore_coarse_none=args.ignore_coarse_none,
            weighting=weighting,
            gradnorm_every=args.gradnorm_every,
        )

        # ── Inline HN mining — mise à jour des poids in-memory ────────────────
        if use_inline_hn and (epoch % args.hn_every == 0):
            hn_res = train_metrics.get("hn_results_by_id")
            if hn_res:
                hn_stats = apply_inline_hn(
                    train_ds,
                    hn_res,
                    boosts=hn_boosts,
                    decay=args.hn_decay,
                    max_weight=args.hn_max_weight,
                    min_weight=args.hn_min_weight,
                )
                print(f"   🔍 HN mining epoch {epoch} : {hn_stats}")

        # Validation avec poids EMA si activé
        if use_ema:
            original_state = ema.apply(model)

        val_metrics = run_epoch(
            val_loader,
            model,
            optimizer,
            device,
            train=False,
            boundary_class_weights=boundary_w,
            coarse_class_weights=coarse_w,
            fine_class_weights=fine_w,
            certainty_class_weights=certainty_w,
            oblique_class_weights=oblique_w,
            role_coarse_class_weights=role_coarse_w,
            verb_family_class_weights=verb_family_w,
            verb_polarity_class_weights=verb_polarity_w,
            verb_aspect_class_weights=verb_aspect_w,
            verb_source_class_weights=verb_source_w,
            lambda_boundary=args.lambda_boundary,
            lambda_coarse=args.lambda_coarse,
            lambda_fine=args.lambda_fine,
            lambda_svo_boundary=args.lambda_svo_boundary,
            lambda_svo=args.lambda_svo,
            lambda_role_coarse=args.lambda_role_coarse,
            lambda_role_oblique=args.lambda_role_oblique,
            lambda_role=args.lambda_role,
            lambda_voice=args.lambda_voice,
            lambda_certainty=args.lambda_certainty,
            lambda_morpho=args.lambda_morpho,
            lambda_verb_ptr=args.lambda_verb_ptr,
            lambda_compat=args.lambda_compat,
            lambda_verb_family=args.lambda_verb_family,
            lambda_verb_family_fine=args.lambda_verb_family_fine,
            lambda_verb_polarity=args.lambda_verb_polarity,
            lambda_verb_aspect=args.lambda_verb_aspect,
            lambda_verb_source=args.lambda_verb_source,
            accum_steps=args.accum_steps,
            log_every=args.log_every,
            focal_gamma=args.focal_gamma,
            max_grad_norm=0.0,   # pas de clipping en eval
            eval_split="val",
            focal_fine_gamma=args.focal_fine_gamma,
            focal_coarse_gamma=args.focal_coarse_gamma,
            focal_role_gamma=args.focal_role_gamma,
            ignore_coarse_none=args.ignore_coarse_none,
            weighting=weighting,
        )

        if use_ema:
            ema.restore(model, original_state)

        # Step scheduler after each epoch
        scheduler.step()

        if args.ner_only_score:
            score = (
                val_metrics["boundary_f1"] * 1.5
                + val_metrics["coarse_macro_f1"] * 1.0
                + val_metrics["fine_macro_f1"] * 2.0
            ) / 4.5
        else:
            # Score composite NER + SVO roles (coarse dérivée + oblique cascade)
            # role_coarse_from_role : F1 sur SUBJ/OBJ/OBLIQ/APPOS via logsumexp (gate fiable)
            # role_oblique_cascaded : F1 sous-types oblique en mode inférence réelle
            score = (
                val_metrics["boundary_f1"]                                      * 1.5
                + val_metrics["coarse_macro_f1"]                                * 1.0
                + val_metrics["fine_macro_f1"]                                  * 2.0
                + val_metrics.get("role_coarse_from_role_macro_f1", 0.0)        * 1.0
                + val_metrics.get("role_oblique_cascaded_macro_f1", 0.0)        * 0.5
            ) / 6.0

        print(f"\n📅 Epoch {epoch}")
        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Boundary F1={train_metrics['boundary_f1']:.4f} | "
            f"Coarse F1={train_metrics['coarse_macro_f1']:.4f} | "
            f"Fine F1={train_metrics['fine_macro_f1']:.4f} | "
            f"Role F1={train_metrics['role_macro_f1']:.4f} | "
            f"Voice F1={train_metrics['voice_macro_f1']:.4f} | "
            f"Certainty F1={train_metrics['certainty_macro_f1']:.4f} | "
            f"Gender F1={train_metrics['gender_macro_f1']:.4f} | "
            f"Number F1={train_metrics['number_macro_f1']:.4f} | "
            f"Person F1={train_metrics['person_macro_f1']:.4f}"
        )
        print(
            f"Val   loss={val_metrics['loss']:.4f} | "
            f"Boundary F1={val_metrics['boundary_f1']:.4f} | "
            f"Coarse F1={val_metrics['coarse_macro_f1']:.4f} | "
            f"Fine F1={val_metrics['fine_macro_f1']:.4f} | "
            f"Role F1={val_metrics['role_macro_f1']:.4f} | "
            f"Voice F1={val_metrics['voice_macro_f1']:.4f} | "
            f"Certainty F1={val_metrics['certainty_macro_f1']:.4f} | "
            f"Gender F1={val_metrics['gender_macro_f1']:.4f} | "
            f"Number F1={val_metrics['number_macro_f1']:.4f} | "
            f"Person F1={val_metrics['person_macro_f1']:.4f} | "
            f"VerbPtr Acc={val_metrics['verb_ptr_acc']:.4f} (n={val_metrics['verb_ptr_n']})"
            f"Score={score:.4f}"
        )

        # ── W&B log epoch ────────────────────────────────────────────────────
        if _wandb_enabled:
            log_dict = {
                "epoch": epoch,
                "score": score,
                "train/loss":           train_metrics["loss"],
                "train/boundary_f1":    train_metrics["boundary_f1"],
                "train/coarse_f1":      train_metrics["coarse_macro_f1"],
                "train/fine_f1":        train_metrics["fine_macro_f1"],
                "train/fine_concrete_f1": train_metrics["fine_concrete_f1"],
                "train/fine_abstract_f1": train_metrics["fine_abstract_f1"],
                "train/svo_boundary_f1": train_metrics["svo_boundary_f1"],
                "train/role_f1":        train_metrics["role_macro_f1"],
                "train/voice_f1":       train_metrics["voice_macro_f1"],
                "train/certainty_f1":   train_metrics["certainty_macro_f1"],
                "train/gender_f1":      train_metrics["gender_macro_f1"],
                "train/number_f1":      train_metrics["number_macro_f1"],
                "train/person_f1":      train_metrics["person_macro_f1"],
                "train/verb_family_f1":      train_metrics.get("verb_family_macro_f1", 0.0),
                "train/verb_family_fine_f1": train_metrics.get("verb_family_fine_macro_f1", 0.0),
                "train/verb_polarity_f1":    train_metrics.get("verb_polarity_macro_f1", 0.0),
                "train/verb_aspect_f1":      train_metrics.get("verb_aspect_macro_f1", 0.0),
                "train/verb_source_f1":      train_metrics.get("verb_source_macro_f1", 0.0),
                "val/loss":             val_metrics["loss"],
                "val/boundary_f1":      val_metrics["boundary_f1"],
                "val/coarse_f1":        val_metrics["coarse_macro_f1"],
                "val/fine_f1":          val_metrics["fine_macro_f1"],
                "val/fine_concrete_f1": val_metrics["fine_concrete_f1"],
                "val/fine_abstract_f1": val_metrics["fine_abstract_f1"],
                "val/svo_boundary_f1":  val_metrics["svo_boundary_f1"],
                "val/role_f1":          val_metrics["role_macro_f1"],
                "val/role_coarse_f1":   val_metrics.get("role_coarse_macro_f1", 0.0),
                "val/role_oblique_f1":  val_metrics.get("role_oblique_macro_f1", 0.0),
                "val/voice_f1":         val_metrics["voice_macro_f1"],
                "val/certainty_f1":     val_metrics["certainty_macro_f1"],
                "val/gender_f1":        val_metrics["gender_macro_f1"],
                "val/number_f1":        val_metrics["number_macro_f1"],
                "val/verb_ptr_acc":     val_metrics["verb_ptr_acc"],
                "val/verb_ptr_n":       val_metrics["verb_ptr_n"],
                "val/verb_family_f1":      val_metrics.get("verb_family_macro_f1", 0.0),
                "val/verb_family_fine_f1": val_metrics.get("verb_family_fine_macro_f1", 0.0),
                "val/verb_polarity_f1":    val_metrics.get("verb_polarity_macro_f1", 0.0),
                "val/verb_aspect_f1":      val_metrics.get("verb_aspect_macro_f1", 0.0),
                "val/verb_source_f1":      val_metrics.get("verb_source_macro_f1", 0.0),
            }
            # Per-label F1 depuis le classification_report (val fine + coarse)
            import re as _re
            # ── Log groupé par famille coarse : val/family_{COARSE}/{hint_xxx}/{f1|prec|rec} ──
            from labels import COARSE_TO_FINE, ID2COARSE
            fine_metrics_by_label = {}  # hint_xxx -> {f1, prec, rec}
            for report_key, prefix in [
                ("fine_report",          "val/fine"),
                ("coarse_report",        "val/coarse"),
                ("role_report",          "val/role"),
                ("role_coarse_report",   "val/role_coarse"),
                ("role_oblique_report",  "val/role_oblique"),
                ("svo_boundary_report",  "val/svo_bnd"),
                ("verb_family_report",   "val/verb_family"),
                ("verb_polarity_report", "val/verb_polarity"),
                ("verb_aspect_report",   "val/verb_aspect"),
                ("verb_source_report",   "val/verb_source"),
            ]:
                report_str = val_metrics.get(report_key, "")
                for line in report_str.splitlines():
                    # ex: "  hint_person_name   0.95  0.88  0.91   1234"
                    m = _re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+", line)
                    if m:
                        lbl, prec, rec, f1 = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
                        log_dict[f"{prefix}_f1_{lbl}"]        = f1
                        log_dict[f"{prefix}_precision_{lbl}"] = prec
                        log_dict[f"{prefix}_recall_{lbl}"]    = rec
                        if prefix == "val/fine":
                            fine_metrics_by_label[lbl] = {"f1": f1, "prec": prec, "rec": rec}
            # ── Log imbriqué val/family_{COARSE}/{hint_xxx}_{metric} ──────────────
            for coarse_id, fine_ids in COARSE_TO_FINE.items():
                coarse_name = ID2COARSE[coarse_id]
                for fine_id in fine_ids:
                    fine_name = FINE_LABELS[fine_id]
                    if fine_name in fine_metrics_by_label:
                        m = fine_metrics_by_label[fine_name]
                        log_dict[f"val/family_{coarse_name}/{fine_name}_f1"]   = m["f1"]
                        log_dict[f"val/family_{coarse_name}/{fine_name}_prec"] = m["prec"]
                        log_dict[f"val/family_{coarse_name}/{fine_name}_rec"]  = m["rec"]
            for item in val_metrics.get("fine_top_confusions", [])[:5]:
                pair = f"{item['true_label']}__{item['pred_label']}"
                log_dict[f"val/fine_confusion_count_{pair}"] = item["count"]
                log_dict[f"val/fine_confusion_row_pct_{pair}"] = item["row_pct"]
            # Log dynamic loss weights
            if not isinstance(weighting, FixedWeighting):
                ramp_lambdas = {
                    "boundary": args.lambda_boundary, "coarse": args.lambda_coarse,
                    "fine": args.lambda_fine, "svo_boundary": args.lambda_svo_boundary,
                    "svo": args.lambda_svo,
                    "voice": args.lambda_voice, "certainty": args.lambda_certainty,
                    "morpho": args.lambda_morpho, "verb_ptr": args.lambda_verb_ptr,
                    "compat": args.lambda_compat,
                }
                eff_weights = weighting.get_effective_weights(ramp_lambdas)
                for k, v in eff_weights.items():
                    log_dict[f"weights/{k}"] = v
            wandb.log(log_dict, step=epoch)

        print("\n[VAL boundary]")
        print(val_metrics["boundary_report"])
        print("[VAL coarse]")
        print(val_metrics["coarse_report"])
        print("[VAL fine]")
        print(val_metrics["fine_report"])
        if val_metrics.get("fine_top_confusions"):
            print("[VAL fine top confusions]")
            for item in val_metrics["fine_top_confusions"][:8]:
                print(f"  {item['true_label']} -> {item['pred_label']}  count={item['count']} row_pct={item['row_pct']:.3f} support={item['support']}")
        if val_metrics.get("fine_hard_labels"):
            print("[VAL fine labels fragiles]")
            for item in val_metrics["fine_hard_labels"][:8]:
                print(f"  {item['label']}  recall={item['recall']:.3f} support={item['support']} top_confused={item['top_confused_with']} ({item['top_confused_count']})")
        if val_metrics.get("fine_confusion_csv"):
            print(f"[VAL fine exports] csv={val_metrics['fine_confusion_csv']} json={val_metrics['fine_diagnostics_json']}")
        print("[VAL svo boundary (verb_trigger)]")
        print(val_metrics["svo_boundary_report"])
        # ⚠️ Ces lignes sont parsées par run_training.py extract_metric() — format EXACT requis
        print(f"Val SVO Bnd F1={val_metrics['svo_boundary_f1']:.4f}")
        # Val Role Crs F1 = role_macro_f1 (12 labels — remplace l'ancienne role_coarse désactivée)
        print(f"Val Role Crs F1={val_metrics.get('role_macro_f1', 0.0):.4f}")
        if val_metrics.get("role_report") and val_metrics["role_report"] != "N/A":
            print("[VAL role (12 labels — SUBJECT/OBJECT/OBLIQUE_*)]")
            print(val_metrics["role_report"])
        if val_metrics.get("verb_family_macro_f1", 0) > 0:
            print(f"[VAL verbfam]  Family F1={val_metrics['verb_family_macro_f1']:.4f}  "
                  f"FamilyFine F1={val_metrics['verb_family_fine_macro_f1']:.4f}  "
                  f"Polarity F1={val_metrics['verb_polarity_macro_f1']:.4f}  "
                  f"Aspect F1={val_metrics['verb_aspect_macro_f1']:.4f}  "
                  f"Source F1={val_metrics['verb_source_macro_f1']:.4f}")
            if val_metrics.get("verb_family_report") and val_metrics["verb_family_report"] != "N/A":
                print("[VAL verb_family]")
                print(val_metrics["verb_family_report"])
        if val_metrics.get("gender_macro_f1", 0) > 0:
            print(f"[VAL morpho]  Gender F1={val_metrics['gender_macro_f1']:.4f}  Number F1={val_metrics['number_macro_f1']:.4f}  Person F1={val_metrics['person_macro_f1']:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_score": best_score,
            "epochs_no_improve": epochs_no_improve,
            "ema_state": ema.state_dict() if use_ema else None,
            "weighting_state": weighting.state_dict() if not isinstance(weighting, FixedWeighting) else None,
        }, "checkpoint_last_multitask.pt")

        if score > best_score + args.min_delta:
            best_score = score
            epochs_no_improve = 0
            # Sauvegarder les poids EMA si activé (meilleure version lissée)
            if use_ema:
                save_state = ema.apply(model)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_score": best_score,
                "ema_state": ema.state_dict() if use_ema else None,
                "weighting_state": weighting.state_dict() if not isinstance(weighting, FixedWeighting) else None,
            }, "checkpoint_best_multitask.pt")
            torch.save(model.state_dict(), "best_model_multitask.pt")
            if use_ema:
                ema.restore(model, save_state)
            print("✅ nouveau best model sauvegardé")
        else:
            epochs_no_improve += 1
            patience_msg = ""
            if args.patience > 0:
                patience_msg = f" [{epochs_no_improve}/{args.patience} sans amélioration]"
            print(f"⏳ Pas d'amélioration du score{patience_msg}")

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"\n🛑 Early stopping déclenché : {epochs_no_improve} epochs sans amélioration "
                f"(patience={args.patience}, min_delta={args.min_delta}). Arrêt à epoch {epoch}."
            )
            break

    print("\n✅ Fin training, évaluation test sur le best model")

    best_ckpt_path = "checkpoint_best_multitask.pt"
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"✅ Best model chargé depuis {best_ckpt_path}")
    else:
        print("⚠️ Pas de checkpoint best trouvé — évaluation avec le modèle courant")

    test_metrics = run_epoch(
        test_loader,
        model,
        optimizer,
        device,
        train=False,
        boundary_class_weights=boundary_w,
        coarse_class_weights=coarse_w,
        fine_class_weights=fine_w,
        certainty_class_weights=certainty_w,
        oblique_class_weights=oblique_w,
        role_coarse_class_weights=role_coarse_w,
        verb_family_class_weights=verb_family_w,
        verb_polarity_class_weights=verb_polarity_w,
        verb_aspect_class_weights=verb_aspect_w,
        verb_source_class_weights=verb_source_w,
        lambda_boundary=args.lambda_boundary,
        lambda_coarse=args.lambda_coarse,
        lambda_fine=args.lambda_fine,
        lambda_svo_boundary=args.lambda_svo_boundary,
        lambda_svo=args.lambda_svo,
        lambda_role_coarse=args.lambda_role_coarse,
        lambda_role_oblique=args.lambda_role_oblique,
        lambda_role=args.lambda_role,
        lambda_voice=args.lambda_voice,
        lambda_certainty=args.lambda_certainty,
        lambda_morpho=args.lambda_morpho,
        lambda_verb_ptr=args.lambda_verb_ptr,
        lambda_compat=args.lambda_compat,
        lambda_verb_family=args.lambda_verb_family,
        lambda_verb_family_fine=args.lambda_verb_family_fine,
        lambda_verb_polarity=args.lambda_verb_polarity,
        lambda_verb_aspect=args.lambda_verb_aspect,
        lambda_verb_source=args.lambda_verb_source,
        accum_steps=args.accum_steps,
        log_every=args.log_every,
        focal_gamma=args.focal_gamma,
        eval_split="test",
        focal_fine_gamma=args.focal_fine_gamma,
        focal_coarse_gamma=args.focal_coarse_gamma,
        focal_role_gamma=args.focal_role_gamma,
        ignore_coarse_none=args.ignore_coarse_none,
        weighting=weighting,
    )

    print("\n🎯 TEST")
    print(f"Loss={test_metrics['loss']:.4f}")
    print(f"Boundary F1={test_metrics['boundary_f1']:.4f}")
    print(f"Coarse   F1={test_metrics['coarse_macro_f1']:.4f}")
    print(f"Fine     F1={test_metrics['fine_macro_f1']:.4f}")
    print(f"SVO Bnd  F1={test_metrics['svo_boundary_f1']:.4f}")
    print(f"Role     F1={test_metrics['role_macro_f1']:.4f}  (12 labels)")
    print(f"Voice    F1={test_metrics['voice_macro_f1']:.4f}")
    print(f"Certainty F1={test_metrics['certainty_macro_f1']:.4f}")
    print(f"Gender   F1={test_metrics['gender_macro_f1']:.4f}")
    print(f"Number   F1={test_metrics['number_macro_f1']:.4f}")
    print(f"Person   F1={test_metrics['person_macro_f1']:.4f}")
    print(f"VerbPtr  Acc={test_metrics['verb_ptr_acc']:.4f} (n={test_metrics['verb_ptr_n']})")
    if test_metrics.get("verb_family_macro_f1", 0) > 0:
        print(f"VerbFam  F1={test_metrics['verb_family_macro_f1']:.4f}  "
              f"FamFine={test_metrics['verb_family_fine_macro_f1']:.4f}  "
              f"Polarity={test_metrics['verb_polarity_macro_f1']:.4f}  "
              f"Aspect={test_metrics['verb_aspect_macro_f1']:.4f}  "
              f"Source={test_metrics['verb_source_macro_f1']:.4f}")

    print("\n[TEST boundary]")
    print(test_metrics["boundary_report"])
    print("[TEST coarse]")
    print(test_metrics["coarse_report"])
    print("[TEST fine]")
    print(test_metrics["fine_report"])
    if test_metrics.get("fine_top_confusions"):
        print("[TEST fine top confusions]")
        for item in test_metrics["fine_top_confusions"][:8]:
            print(f"  {item['true_label']} -> {item['pred_label']}  count={item['count']} row_pct={item['row_pct']:.3f} support={item['support']}")
    if test_metrics.get("fine_hard_labels"):
        print("[TEST fine labels fragiles]")
        for item in test_metrics["fine_hard_labels"][:8]:
            print(f"  {item['label']}  recall={item['recall']:.3f} support={item['support']} top_confused={item['top_confused_with']} ({item['top_confused_count']})")
    if test_metrics.get("fine_confusion_csv"):
        print(f"[TEST fine exports] csv={test_metrics['fine_confusion_csv']} json={test_metrics['fine_diagnostics_json']}")
    print("[TEST svo boundary (verb_trigger)]")
    print(test_metrics["svo_boundary_report"])
    if test_metrics.get("role_report") and test_metrics["role_report"] != "N/A":
        print("[TEST role (12 labels)]")
        print(test_metrics["role_report"])
    if test_metrics.get("verb_family_report") and test_metrics["verb_family_report"] != "N/A":
        print("[TEST verb_family]")
        print(test_metrics["verb_family_report"])
    if test_metrics.get("gender_macro_f1", 0) > 0:
        print(f"[TEST morpho]  Gender F1={test_metrics['gender_macro_f1']:.4f}  Number F1={test_metrics['number_macro_f1']:.4f}  Person F1={test_metrics['person_macro_f1']:.4f}")

    # ── W&B log test final ───────────────────────────────────────────────────
    if _wandb_enabled:
        wandb.log({
            "test/boundary_f1":  test_metrics["boundary_f1"],
            "test/coarse_f1":    test_metrics["coarse_macro_f1"],
            "test/fine_f1":          test_metrics["fine_macro_f1"],
            "test/fine_concrete_f1": test_metrics["fine_concrete_f1"],
            "test/fine_abstract_f1": test_metrics["fine_abstract_f1"],
            "test/svo_boundary_f1": test_metrics["svo_boundary_f1"],
            "test/role_f1":      test_metrics["role_macro_f1"],
            "test/voice_f1":     test_metrics["voice_macro_f1"],
            "test/certainty_f1": test_metrics["certainty_macro_f1"],
            "test/gender_f1":    test_metrics["gender_macro_f1"],
            "test/number_f1":    test_metrics["number_macro_f1"],
            "test/loss":         test_metrics["loss"],
            "test/verb_family_f1":      test_metrics.get("verb_family_macro_f1", 0.0),
            "test/verb_family_fine_f1": test_metrics.get("verb_family_fine_macro_f1", 0.0),
            "test/verb_polarity_f1":    test_metrics.get("verb_polarity_macro_f1", 0.0),
            "test/verb_aspect_f1":      test_metrics.get("verb_aspect_macro_f1", 0.0),
            "test/verb_source_f1":      test_metrics.get("verb_source_macro_f1", 0.0),
            **{
                f"test/fine_confusion_count_{item['true_label']}__{item['pred_label']}": item["count"]
                for item in test_metrics.get("fine_top_confusions", [])[:5]
            },
        })
        wandb.finish()
        print("📊 W&B run terminé")


if __name__ == "__main__":
    main()
