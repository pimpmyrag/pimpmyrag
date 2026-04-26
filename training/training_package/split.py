import json
import random

random.seed(42)

lines = [json.loads(l) for l in open("dataset_corrected_v1_checked.jsonl", "r")]

random.shuffle(lines)

n = len(lines)
train = lines[:int(0.8*n)]
val = lines[int(0.8*n):int(0.9*n)]
test = lines[int(0.9*n):]

def dump(name, data):
    with open(name, "w") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

dump("train.jsonl", train)
dump("val.jsonl", val)
dump("test.jsonl", test)