"""探测编译后 xmodel 的输入输出格式，确认板端 int8/float 口径。"""
import xir

P = "/workspace/mymodel/gate320/compiled_E/gate320_E.xmodel"
g = xir.Graph.deserialize(P)
root = g.get_root_subgraph()


def attr(o, k):
    return o.get_attr(k) if o.has_attr(k) else None


def dt(t):
    try:
        return t.get_data_type()
    except Exception:
        return getattr(t, "dtype", "?")


def show(tag, tensors):
    print("=== %s ===" % tag)
    for t in tensors:
        print("  %-30s shape=%-16s dtype=%-8s fix_point=%s"
              % (t.name[:30], list(t.dims), dt(t), attr(t, "fix_point")))


show("root 输入", root.get_input_tensors())
show("root 输出", root.get_output_tensors())

print("=== 直接子图 ===")
try:
    children = root.toposort_child_subgraph()
except Exception as e:
    print("  toposort_child_subgraph 不可用:", e)
    children = root.toposort()

ndpu = 0
for s in children:
    d = attr(s, "device")
    if d and str(d).upper() == "DPU":
        ndpu += 1
        print("  DPU 子图: %s" % s.get_name())
        for t in s.get_input_tensors():
            print("     in  %-24s shape=%-14s dtype=%-8s fix=%s"
                  % (t.name[:24], list(t.dims), dt(t), attr(t, "fix_point")))
        for t in s.get_output_tensors():
            print("     out %-24s shape=%-14s dtype=%-8s fix=%s"
                  % (t.name[:24], list(t.dims), dt(t), attr(t, "fix_point")))
print("  直接子图 %d 个，其中 DPU %d 个" % (len(children), ndpu))
