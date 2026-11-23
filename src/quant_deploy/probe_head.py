import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    o = d["output"]
    keys = [k for k in o if "heads" in k or "DetHead" in k
            or "dequant" in k.lower()]
    print("=== %s ===" % p)
    print("head/dequant 节点数:", len(keys))
    for k in keys:
        print("   ", o[k], "|", k[-95:])
    print()
