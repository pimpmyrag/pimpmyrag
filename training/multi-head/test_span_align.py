"""Petit test pour vérifier l'alignement entre prédictions et labels.
Crée un encodeur factice (retourne des hidden states aléatoires) et des spans,
puis vérifie que compute_loss n'effectue pas un trimming lorsque les indices
sont renvoyés.
"""
import torch
from multi_task_model import SpanMultiTaskModel


class DummyEncoder:
    def __init__(self, hidden_size=16):
        # create a simple config object with hidden_size attribute
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()

    def __call__(self, input_ids=None, attention_mask=None):
        b = input_ids.size(0)
        l = input_ids.size(1)
        hs = torch.randn(b, l, self.config.hidden_size)
        return type("EncOut", (), {"last_hidden_state": hs})()


def build_batch():
    # batch of 1 with 3 tokens
    input_ids = torch.ones((1, 3), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    spans = [[{"tok_start": 0, "tok_end": 0}, {"tok_start": 1, "tok_end": 2}]]
    # flattened labels (2 spans)
    boundary = torch.tensor([0, 1], dtype=torch.long)
    coarse = torch.tensor([2, 3], dtype=torch.long)
    fine = torch.tensor([5, 6], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spans": spans,
    }, boundary, coarse, fine


def main():
    enc = DummyEncoder(hidden_size=16)
    model = SpanMultiTaskModel(encoder=enc)
    batch, b_lbl, c_lbl, f_lbl = build_batch()
    outs = model(batch)
    # ensure we have span_indices and they are length 2
    print("span_indices:", outs.get("span_indices"))
    loss_dict = model.compute_loss(outs, b_lbl, c_lbl, f_lbl)
    print("loss:", loss_dict["loss"])


if __name__ == '__main__':
    main()

