import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_dir = "outputs/xml_ner_bilou"
onnx_path = "outputs/xml_ner_bilou/xmlr_bilou.onnx"

# Load
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained(model_dir)
model.eval()

# Dummy input
dummy = tokenizer(
    ["Ceci est un test ."],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=256,
)

# Export
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
    do_constant_folding=True,
)
tokenizer.save_pretrained("outputs/xmlr_ner_bilou/tokenizer_full")

print("✅ Export ONNX terminé:", onnx_path)