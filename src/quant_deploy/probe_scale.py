import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "qr_splitlm_probe/quant_info.json"
d = json.load(open(p))
print("top keys:", list(d.keys())[:10])


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + "/%d" % i)
    else:
        yield path, o


hits = [(pp, v) for pp, v in walk(d) if "dequant" in pp.lower()]
print("dequant 条目数:", len(hits))
for pp, v in hits[:24]:
    print("  ", pp[-78:], "=", v)
