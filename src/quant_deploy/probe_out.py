import json
import sys

p = sys.argv[1]
d = json.load(open(p))
o = d.get("output")
print("output type:", type(o).__name__)
if isinstance(o, dict):
    print("output keys:", list(o.keys())[:12])
elif isinstance(o, list):
    print("output len:", len(o))
    for i, it in enumerate(o[:6]):
        print("---", i, "---")
        print(json.dumps(it)[:600])
print("=== 全量 output 前 3000 字符 ===")
print(json.dumps(o)[:3000])
