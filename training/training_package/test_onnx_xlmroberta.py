from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
tok.save_pretrained("deberta/tokenizer_export")